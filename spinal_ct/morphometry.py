"""Editable body measurement sessions, review invalidation and reproducible export."""
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import csv
import hashlib
import html
import json
from pathlib import Path
import tempfile

import numpy as np
import SimpleITK as sitk

from spinal_ct.body_region import BodyPlane, binary_mask, sample_plane, spacing_xy
from spinal_ct.blob_analysis import analyze_blob, normalize_region
from spinal_ct.area_monte_carlo import estimate_area, validate_sampling
from spinal_ct.localization import LEVELS
from spinal_ct.review import ct_fingerprint

LANDMARK_NAMES = ('anterior_superior', 'anterior_inferior', 'middle_superior',
                  'middle_inferior', 'posterior_superior', 'posterior_inferior')


@dataclass(frozen=True)
class BodyAutoParams:
    anterior_fraction: float = 0.65
    neck_ratio: float = 0.60
    ant_ref_span: float = 0.45
    search_start: float = 0.35
    search_end: float = 0.85
    min_component_ratio: float = 0.15
    inset_mm: float = 2.5
    band_half_mm: float = 1.5
    min_hu: float | None = None
    min_normal_ap_ratio: float = 0.80
    max_normal_ap_ratio: float = 1.30

    def __post_init__(self):
        if not np.isfinite(self.anterior_fraction) or not (0.35 <= self.anterior_fraction <= 1.0):
            raise ValueError('anterior_fraction must be between 0.35 and 1.0.')
        if not np.isfinite(self.neck_ratio) or not (0.10 <= self.neck_ratio <= 0.95):
            raise ValueError('neck_ratio must be between 0.10 and 0.95.')
        if not np.isfinite(self.ant_ref_span) or self.ant_ref_span <= 0:
            raise ValueError('ant_ref_span must be positive.')
        if not np.isfinite(self.search_start) or not np.isfinite(self.search_end) or self.search_start >= self.search_end:
            raise ValueError('search_start must be less than search_end.')
        if not np.isfinite(self.min_component_ratio) or self.min_component_ratio <= 0:
            raise ValueError('min_component_ratio must be positive.')
        if not np.isfinite(self.inset_mm) or self.inset_mm < 0:
            raise ValueError('inset_mm must be non-negative.')
        if not np.isfinite(self.band_half_mm) or self.band_half_mm <= 0:
            raise ValueError('band_half_mm must be positive.')
        if self.min_hu is not None and not np.isfinite(self.min_hu):
            raise ValueError('min_hu must be finite when specified.')
        if not np.isfinite(self.min_normal_ap_ratio) or not np.isfinite(self.max_normal_ap_ratio) or self.min_normal_ap_ratio >= self.max_normal_ap_ratio:
            raise ValueError('Invalid normal AP ratio range.')

    def to_dict(self) -> dict:
        return asdict(self)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill internal 2D holes in a boolean mask without modifying input in place."""
    clean = binary_mask(mask).copy()
    if not clean.any():
        return clean
    filled = sitk.BinaryFillhole(sitk.GetImageFromArray(clean.astype(np.uint8)), True)
    return sitk.GetArrayFromImage(filled).astype(bool)


def _largest_anterior_component(mask: np.ndarray, min_component_ratio: float = 0.15) -> np.ndarray:
    """Select the anterior vertebral body component and discard posterior arch fragments."""
    clean = binary_mask(mask).copy()
    if not clean.any():
        return clean
    cc_img = sitk.ConnectedComponent(sitk.GetImageFromArray(clean.astype(np.uint8)), True)
    labels = sitk.GetArrayFromImage(cc_img)
    max_label = int(labels.max())
    if max_label <= 1:
        return clean
    counts = np.bincount(labels.ravel())[1:]
    max_area = int(counts.max())
    candidates = []
    for label_id, area in enumerate(counts, start=1):
        if area >= max(3, int(min_component_ratio * max_area)):
            rows, cols = np.nonzero(labels == label_id)
            candidates.append((float(cols.min()), float(cols.mean()), -int(area), label_id))
    candidates.sort()
    return labels == candidates[0][3]


def auto_segment_body(ct: np.ndarray, valid: np.ndarray, hint: np.ndarray,
                      spacing: tuple, *, params: BodyAutoParams | None = None) -> np.ndarray:
    """Isolate the 2D vertebral body mask from whole-vertebra sagittal hint and CT."""
    params = params or BodyAutoParams()
    sx, sy = spacing_xy(spacing)
    ct_arr = np.asarray(ct, dtype=float)
    valid_mask = binary_mask(valid).copy()
    hint_mask = binary_mask(hint).copy()
    if ct_arr.shape != valid_mask.shape or ct_arr.shape != hint_mask.shape:
        raise ValueError('CT, valid FOV and hint masks must share the same 2D shape.')
    if not np.isfinite(params.anterior_fraction) or not 0.35 <= params.anterior_fraction <= 1.0:
        raise ValueError('anterior_fraction must be between 0.35 and 1.0.')
    if not np.isfinite(params.neck_ratio) or not 0.10 <= params.neck_ratio <= 0.95:
        raise ValueError('neck_ratio must be between 0.10 and 0.95.')

    candidate = hint_mask & valid_mask
    if params.min_hu is not None:
        if not np.isfinite(params.min_hu):
            raise ValueError('min_hu must be finite when specified.')
        candidate = candidate & (ct_arr >= float(params.min_hu))

    if not candidate.any():
        raise ValueError('No vertebral candidate pixels found within the valid CT field of view.')

    candidate = _fill_holes(candidate) & valid_mask
    body = _largest_anterior_component(candidate, params.min_component_ratio)

    cols = np.flatnonzero(body.any(axis=0))
    if len(cols) >= 6:
        c_min, c_max = int(cols[0]), int(cols[-1])
        span = c_max - c_min + 1
        col_heights = body[:, c_min:c_max + 1].sum(axis=0).astype(float) * sy
        ant_span = max(2, int(round(params.ant_ref_span * span)))
        ant_ref_height = float(np.percentile(col_heights[:ant_span], 75))

        s_start = max(2, int(round(params.search_start * span)))
        s_end = min(span - 1, max(s_start + 1, int(round(params.search_end * span))))
        if ant_ref_height > 0 and s_end > s_start:
            window = col_heights[s_start:s_end]
            min_idx_rel = int(np.argmin(window))
            valley_height = float(window[min_idx_rel])
            if valley_height <= params.neck_ratio * ant_ref_height:
                drop_indices = np.flatnonzero(window <= params.neck_ratio * ant_ref_height)
                neck_rel = int(drop_indices[0]) if len(drop_indices) else min_idx_rel
                cut_col = c_min + min(s_start + min_idx_rel,
                                      max(s_start + neck_rel, int(round(params.anterior_fraction * span))))
                trimmed = body.copy()
                trimmed[:, cut_col:] = False
                if trimmed.any():
                    body = _largest_anterior_component(trimmed, params.min_component_ratio)

    body = _fill_holes(body) & valid_mask
    if int(body.sum()) < 3:
        raise ValueError('Segmented vertebral body region is too small (< 3 pixels).')
    return body


def _wall_x(mask: np.ndarray, sx: float) -> tuple[float, float]:
    """Anterior/posterior wall x (mm) from the middle half of body height."""
    rows = np.flatnonzero(mask.any(axis=1))
    if len(rows) < 2:
        raise ValueError('Mask needs at least 2 occupied rows to estimate anterior/posterior walls.')
    lo = len(rows) // 4
    hi = max(lo + 1, (3 * len(rows)) // 4)
    mid = rows[lo:hi]
    first = [int(np.flatnonzero(mask[r])[0]) for r in mid]
    last = [int(np.flatnonzero(mask[r])[-1]) for r in mid]
    return float(np.median(first) * sx), float(np.median(last) * sx)


def _band_edges(mask: np.ndarray, sx: float, sy: float, x_mm: float, half_mm: float = 1.5) -> tuple[float, float]:
    """Median superior/inferior contour y (mm) in a vertical band, filtering staircase edge steps."""
    occupied_cols = np.flatnonzero(mask.any(axis=0))
    eff_half = max(float(half_mm), 0.6 * float(sx))
    cols = [int(c) for c in occupied_cols if abs(c * sx - x_mm) <= eff_half]
    if not cols:
        raise ValueError('No mask pixels near landmark column.')
    ys = [np.flatnonzero(mask[:, c]) for c in cols]
    heights = np.array([r[-1] - r[0] for r in ys])
    keep = heights >= np.median(heights)
    ys = [r for r, k in zip(ys, keep) if k]
    return float(np.median([r[-1] for r in ys]) * sy), float(np.median([r[0] for r in ys]) * sy)


def auto_detect_landmarks(mask: np.ndarray, plane: BodyPlane, *,
                          params: BodyAutoParams | None = None) -> list[list[float]]:
    """Detect 6 landmarks along vertical columns inset from the anterior and posterior walls."""
    from spinal_ct.body_region import spacing_xy
    params = params or BodyAutoParams()
    clean = binary_mask(mask).copy()
    if tuple(plane.size_xy[::-1]) != clean.shape:
        raise ValueError('Mask shape does not match plane dimensions.')
    cc_img = sitk.ConnectedComponent(sitk.GetImageFromArray(clean.astype(np.uint8)), True)
    if int(sitk.GetArrayViewFromImage(cc_img).max()) > 1:
        raise ValueError('Mask contains multiple disconnected components.')
    sx, sy = spacing_xy(plane.spacing_xy_mm)
    xa, xp = _wall_x(clean, sx)
    width_mm = xp - xa
    if width_mm <= 2 * sx:
        raise ValueError('Vertebral body mask is too narrow to place anterior/middle/posterior landmarks.')
    eff_inset = min(float(params.inset_mm), 0.25 * width_mm)
    pts: list[list[float]] = []
    for x in (xa + eff_inset, 0.5 * (xa + xp), xp - eff_inset):
        sup, inf = _band_edges(clean, sx, sy, x, half_mm=params.band_half_mm)
        pts.extend([[float(x), float(sup)], [float(x), float(inf)]])
    measure_heights(pts, plane)
    return pts


def mask_digest(mask):
    return hashlib.sha256(np.asarray(mask, dtype=np.uint8).tobytes()).hexdigest()


def measure_heights(points, plane):
    if not points:
        return None
    points = np.asarray(points, float)
    if points.shape != (6, 2) or not np.isfinite(points).all():
        raise ValueError('Height measurements need all six finite landmarks.')
    spacing = np.asarray(plane.spacing_xy_mm)
    if (points < -spacing/2).any() or (points > (np.array(plane.size_xy)-.5)*spacing).any():
        raise ValueError('Height landmarks extend beyond the plane.')
    paired = points.reshape(3, 2, 2)
    if (paired[:, 0, 1] <= paired[:, 1, 1]).any() or (np.diff(paired.mean(1)[:, 0]) <= 0).any():
        raise ValueError('Landmarks must be superior/inferior pairs in anterior → middle → posterior order.')
    heights = np.linalg.norm(paired[:, 0]-paired[:, 1], axis=1)
    return dict(anterior_mm=float(heights[0]), middle_mm=float(heights[1]), posterior_mm=float(heights[2]),
                anterior_posterior_ratio=float(heights[0]/heights[2]),
                middle_posterior_ratio=float(heights[1]/heights[2]),
                definition='Euclidean distances between manually confirmed superior/inferior landmark pairs.')


@dataclass
class BodySession:
    input_fingerprint: str
    level: str
    source_kind: str
    plane: BodyPlane
    ct: np.ndarray
    valid: np.ndarray
    hint: np.ndarray
    mask: np.ndarray | None = None
    landmarks: list = field(default_factory=list)
    age_group: str = 'unspecified'
    candidates: dict | None = None
    sampling_config: dict = field(default_factory=lambda: dict(samples=100_000, seed=42))
    auto_provenance: dict | None = None
    _approved_signature: str | None = None
    result: dict | None = None

    def __post_init__(self):
        shape = tuple(self.plane.size_xy[::-1])
        if (self.level not in LEVELS or self.source_kind not in ('Baseline', 'Consensus')
                or self.age_group not in ('unspecified', 'adult', 'pediatric')
                or self.ct.shape != shape or self.valid.shape != shape or self.hint.shape != shape
                or not np.isfinite(self.ct).all()):
            raise ValueError('Invalid body measurement session.')
        if self.mask is None:
            self.mask = np.zeros(shape, bool)
        self.set_mask(self.mask, clear_landmarks=False)

    def _signature(self):
        details = dict(plane=self.plane.to_dict(), level=self.level, source_kind=self.source_kind,
                       fingerprint=self.input_fingerprint, landmarks=self.landmarks, age_group=self.age_group)
        return hashlib.sha256((mask_digest(self.mask)+json.dumps(details, sort_keys=True)).encode()).hexdigest()

    @property
    def reviewed(self):
        return self._approved_signature is not None and self._approved_signature == self._signature()

    def invalidate(self):
        self._approved_signature = None
        self.result = None

    def set_candidates(self, detection):
        # LoG is ancillary to area/height measurements on the unchanged body mask.
        self.candidates = detection
        if self.result is not None:
            self.result['blob_detection'] = detection

    def set_mask(self, mask, *, clear_landmarks=True):
        value = binary_mask(mask)
        if value.shape != self.ct.shape or (value & ~self.valid).any():
            raise ValueError('Body mask must stay within the sampled CT field of view.')
        self.mask = value.copy()
        if clear_landmarks:
            self.landmarks = []
        self.auto_provenance = None
        self.invalidate()

    def set_landmarks(self, points):
        points = np.asarray(points, float)
        if points.size == 0:
            self.landmarks = []
        elif points.ndim == 2 and points.shape[1] == 2 and len(points) <= 6 and np.isfinite(points).all():
            lower = -np.asarray(self.plane.spacing_xy_mm)/2
            upper = (np.asarray(self.plane.size_xy)-.5)*self.plane.spacing_xy_mm
            if (points < lower).any() or (points > upper).any():
                raise ValueError('Landmark lies outside the plane.')
            self.landmarks = points.tolist()
        else:
            raise ValueError('Expected up to six finite plane landmarks.')
        self.auto_provenance = None
        self.invalidate()

    def auto_populate(self, *, params: BodyAutoParams | None = None, approve: bool = False):
        params = params or BodyAutoParams()
        mask = auto_segment_body(self.ct, self.valid, self.hint, self.plane.spacing_xy_mm, params=params)
        landmarks = auto_detect_landmarks(mask, self.plane, params=params)
        self.set_mask(mask, clear_landmarks=True)
        self.set_landmarks(landmarks)
        self.auto_provenance = dict(method='auto_v1', params=params.to_dict())
        if approve:
            self.approve()

    def approve(self):
        features = analyze_blob(self.mask, self.plane.spacing_xy_mm)
        if len(self.landmarks) not in (0, 6):
            raise ValueError('Finish all six height landmarks or clear them before review.')
        measure_heights(self.landmarks, self.plane)
        if self.landmarks:
            indices = np.floor(np.array(self.landmarks)/self.plane.spacing_xy_mm+.5).astype(int)
            indices = np.clip(indices, [0, 0], np.array(self.plane.size_xy)-1)
            if not self.valid[indices[:, 1], indices[:, 0]].all():
                raise ValueError('Height landmarks must lie within the CT field of view.')
        if features['pixel_count'] < 3:
            raise ValueError('Region is too small to confirm.')
        self._approved_signature = self._signature()
        self.result = None

    def measure(self, *, samples=100_000, seed=42):
        if not self.reviewed:
            raise ValueError('Confirm the current body region and plane before measuring.')
        configuration = validate_sampling(samples, seed)
        features = analyze_blob(self.mask, self.plane.spacing_xy_mm)
        warnings = list(features['warnings'])
        params = (BodyAutoParams(**self.auto_provenance['params'])
                  if self.auto_provenance and 'params' in self.auto_provenance
                  else BodyAutoParams())
        heights = measure_heights(self.landmarks, self.plane)
        if heights is None:
            warnings.append('height_landmarks_missing')
        else:
            if not (params.min_normal_ap_ratio <= heights['anterior_posterior_ratio'] <= params.max_normal_ap_ratio):
                warnings.append('abnormal_ap_height_ratio')
            if heights['middle_posterior_ratio'] < params.min_normal_ap_ratio:
                warnings.append('low_middle_posterior_ratio')
        if self.age_group == 'unspecified':
            warnings.append('age_group_unspecified')
        eroded_valid = self.valid.copy()
        for axis in (0, 1):
            for shift in (-1, 1):
                eroded_valid &= np.roll(self.valid, shift, axis=axis)
        if (self.mask & ~eroded_valid).any():
            warnings.append('touches_ct_field_boundary')
        normalization = normalize_region(self.ct, self.mask, self.plane.spacing_xy_mm, features)
        self.result = dict(schema_version=1, task='vertebral_body_morphometry',
                           input_fingerprint=self.input_fingerprint, level=self.level, source_kind=self.source_kind,
                           review_status='confirmed_by_user', review_required=True,
                           status='needs_review' if warnings else 'measured',
                           region_signature=self._signature(), mask_sha256=mask_digest(self.mask),
                           age_group=self.age_group, plane=self.plane.to_dict(),
                           features=features, heights=heights,
                           landmarks_xy_mm=self.landmarks, landmark_names=list(LANDMARK_NAMES),
                           normalization=normalization['transform'], warnings=warnings,
                           auto_provenance=self.auto_provenance,
                           monte_carlo_area=estimate_area(self.mask, self.plane.spacing_xy_mm,
                                                         samples=samples, seed=seed, features=features),
                           blob_detection=self.candidates,
                           interpretation='Shape measurements on a user-defined body region; no diagnostic or AO classification.')
        self.sampling_config = configuration
        return self.result


def save_session(session, output, *, samples=100_000, seed=42):
    """Publish a complete new folder. Drafts retain their unconfirmed status."""
    output = Path(output)
    if output.exists():
        raise ValueError('Output already exists; choose a new directory.')
    if not session.mask.any():
        raise ValueError('Draw a region before saving.')
    configuration = validate_sampling(samples, seed)
    report = session.measure(samples=samples, seed=seed) if session.reviewed else None
    document = dict(schema_version=1, task='vertebral_body_region',
                    input_fingerprint=session.input_fingerprint, level=session.level,
                    source_kind=session.source_kind, plane=session.plane.to_dict(), age_group=session.age_group,
                    landmarks_xy_mm=session.landmarks, mask='body_mask.npy',
                    mask_sha256=mask_digest(session.mask),
                    auto_provenance=session.auto_provenance,
                    region_signature=session._signature(), reviewed=session.reviewed,
                    blob_detection_config=(session.candidates or {}).get('config'),
                    area_sampling=configuration,
                    body_definition='User outline of the entire body cross-section, including its enclosed interior.',
                    created_utc=datetime.now(timezone.utc).isoformat())
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.body-', dir=output.parent) as temporary:
        root = Path(temporary)/'result'
        root.mkdir()
        np.save(root/'body_mask.npy', session.mask.astype(np.uint8), allow_pickle=False)
        (root/'region.json').write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        if report:
            (root/'measurements.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
            normalized = normalize_region(session.ct, session.mask, session.plane.spacing_xy_mm, report['features'])
            np.save(root/'normalized_ct.npy', normalized['ct'], allow_pickle=False)
            np.save(root/'normalized_mask.npy', normalized['mask'].astype(np.uint8), allow_pickle=False)
            flat = dict(level=session.level, source_kind=session.source_kind, age_group=session.age_group,
                        area_mm2=report['features']['area_mm2'], fill_ratio=report['features']['fill_ratio'],
                        solidity=report['features']['solidity'], height_width_ratio=report['features']['height_width_ratio'],
                        **{key: (report['heights'] or {}).get(key) for key in (
                            'anterior_mm', 'middle_mm', 'posterior_mm',
                            'anterior_posterior_ratio', 'middle_posterior_ratio')},
                        mc_area_mm2=report['monte_carlo_area']['area_mm2'],
                        mc_elbow_area_mm2=(report['monte_carlo_area'].get('elbow') or {}).get('area_mm2'),
                        mc_elbow_samples=(report['monte_carlo_area'].get('elbow') or {}).get('samples'),
                        mc_samples=samples, mc_seed=seed, warnings=';'.join(report['warnings']))
            with (root/'features.csv').open('w', newline='', encoding='utf-8-sig') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(flat))
                writer.writeheader()
                writer.writerow(flat)
            from spinal_ct.morphometry_render import export_preview
            export_preview(session, report, root/'preview.png')
            rows = ''.join(f'<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>' for k, v in flat.items())
            (root/'report.html').write_text('<!doctype html><html lang="vi"><meta charset="utf-8">'
                '<title>Đo thân đốt sống</title><style>body{font:16px sans-serif;max-width:1200px;margin:30px auto}'
                'th,td{padding:8px;text-align:left;border-bottom:1px solid #ccc}img{width:100%}</style>'
                '<h1>Đo hình dạng thân đốt sống</h1><p>Vùng đo được người dùng xác nhận; kết quả cần kiểm tra. '
                'Khoảng Monte Carlo chỉ mô tả sai số lấy mẫu trên mask.</p><table>'+rows+
                '</table><img src="preview.png" alt="Vùng đo, chuẩn hóa và hội tụ diện tích"></html>', encoding='utf-8')
        root.rename(output)
    return output


def load_session(folder, source, *, fingerprint=None):
    folder = Path(folder)
    document = json.loads((folder/'region.json').read_text(encoding='utf-8'))
    fingerprint = fingerprint or ct_fingerprint(source)
    if (document.get('schema_version') != 1 or document.get('task') != 'vertebral_body_region'
            or document.get('input_fingerprint') != fingerprint or document.get('mask') != 'body_mask.npy'):
        raise ValueError('Region schema or CT fingerprint mismatch.')
    plane = BodyPlane(**document['plane'])
    ct, valid, hint = sample_plane(source, plane)
    mask = binary_mask(np.load(folder/'body_mask.npy', allow_pickle=False))
    if mask_digest(mask) != document.get('mask_sha256'):
        raise ValueError('Body mask has changed since this region was saved.')
    session = BodySession(fingerprint, document['level'], document['source_kind'], plane, ct, valid, hint,
                          mask=mask, age_group=document['age_group'])
    session.set_landmarks(document['landmarks_xy_mm'])
    session.auto_provenance = document.get('auto_provenance')
    if session._signature() != document.get('region_signature'):
        raise ValueError('Region geometry, landmarks or provenance changed since saving.')
    if document.get('reviewed') is True:
        session.approve()
    configuration = document.get('area_sampling')
    if configuration is None and (folder/'measurements.json').is_file():
        # Older schema-1 regions stored sampling settings only in the report.
        previous = json.loads((folder/'measurements.json').read_text(encoding='utf-8'))
        if previous.get('region_signature') == session._signature():
            mc = previous.get('monte_carlo_area', {})
            configuration = dict(samples=mc.get('samples'), seed=mc.get('seed'))
    if configuration is not None:
        session.sampling_config = validate_sampling(**configuration)
    if document.get('blob_detection_config'):
        from spinal_ct.blob_detection import detect_blobs
        session.set_candidates(detect_blobs(session.ct, session.plane.spacing_xy_mm, session.valid,
                                             **document['blob_detection_config']))
    # Measurements are recomputed from the current data, not trusted from a saved report.
    return session
