"""AO Spine 2-Step Vertebra Prototype Bank (Ngân hàng mẫu đốt sống 2 bước).

Architecture:
- Step 1: Normal vertebra prototypes by spinal level (T10-L5) to detect anomalies / compute p_anomaly.
- Step 2: Lesion prototypes for AO Spine classification (A1, A2, A3, A4, B2, B3, C) to identify fracture type.

Zero Patient Leakage & Strict Fold Isolation:
- Prototype centroids, level-wise normal distributions, and Z-score statistics (mean, std)
  are strictly fitted ONLY on the training set of each fold.
- Evaluation is performed on the validation set of that fold with mathematical zero leakage guarantee.
- Rare challenge patients (A1, B3) are held out strictly for zero-shot testing and never enter training.
"""
from typing import Any
from pathlib import Path
import json
import numpy as np

SPINE_LEVELS = ['T10', 'T11', 'T12', 'L1', 'L2', 'L3', 'L4', 'L5']
ALL_AOSPINE_CLASSES = ['A1', 'A2', 'A3', 'A4', 'B2', 'B3', 'C']


def extract_roi_prototype_features(
    body_roi: np.ndarray,
    arch_roi: np.ndarray | None = None,
    morph_metrics: dict[str, Any] | None = None,
) -> np.ndarray:
    """Extract standard morphometric and radiographic concept features from 2.5D Dual-ROI arrays.

    Parameters
    ----------
    body_roi : np.ndarray
        5-channel Body ROI array of shape (5, 128, 128) in range [0.0, 1.0].
        Channels:
        - Ch 0: Mid-sagittal slice with Bone Window (WL=400, WW=1800)
        - Ch 1: Parasagittal slice at -4.0 mm (Bone Window)
        - Ch 2: Parasagittal slice at +4.0 mm (Bone Window)
        - Ch 3: Axial slice across superior one-third of vertebral body (Bone Window)
        - Ch 4: Body + Canal mask (values in [0.0, 1.0])
    arch_roi : np.ndarray, optional
        5-channel Arch ROI array of shape (5, 128, 128) in range [0.0, 1.0].
    morph_metrics : dict, optional
        Optional geometric morphometry metadata to concatenate:
        `ha_hp_ratio`, `hm_hp_ratio`, `relative_collapse_ratio`, `delta_posterior_mm`, `isp_ratio`, `translation_mm`.

    Returns
    -------
    np.ndarray
        16-dimensional (or 22-dimensional if morph_metrics is provided) float32 feature vector,
        L2-normalized.
    """
    body_roi = np.asarray(body_roi, dtype=np.float32)
    ch0_bone = body_roi[0]
    ch1_para_l = body_roi[1]
    ch2_para_r = body_roi[2]
    ch3_axial = body_roi[3]
    ch4_mask = body_roi[4]

    # Body mask: Ch 4 >= 0.95
    body_mask = (ch4_mask >= 0.95)
    # Canal zone: 0.40 <= Ch 4 < 0.95
    canal_mask = (ch4_mask >= 0.40) & (ch4_mask < 0.95)

    # 1. Body bone density (mean, std, 90th percentile)
    if body_mask.any():
        body_voxels = ch0_bone[body_mask]
        f_density_mean = float(np.mean(body_voxels))
        f_density_std = float(np.std(body_voxels))
        f_density_p90 = float(np.percentile(body_voxels, 90))
    else:
        f_density_mean = 0.5
        f_density_std = 0.1
        f_density_p90 = 0.6

    # 2. Parasagittal asymmetry (fracture comminution or sagittal tilt)
    f_para_asym = float(np.mean(np.abs(ch1_para_l - ch2_para_r)))

    # 3. Canal encroachment (burst fracture into spinal canal)
    if canal_mask.any():
        canal_voxels = ch0_bone[canal_mask]
        f_canal_mean = float(np.mean(canal_voxels))
        # Fraction of canal encroached by bone (HU > 0.50 in bone window)
        f_canal_bone_ratio = float(np.mean(canal_voxels >= 0.50))
    else:
        f_canal_mean = 0.0
        f_canal_bone_ratio = 0.0

    # 4. Geometric aspect ratio and wedge deformity
    occ_rows, occ_cols = np.where(body_mask)
    if len(occ_rows) >= 10:
        h_px = float(occ_rows.max() - occ_rows.min() + 1)
        w_px = float(occ_cols.max() - occ_cols.min() + 1)
        f_aspect_ratio = float(np.clip(h_px / max(w_px, 1.0), 0.1, 2.0))

        # Anterior vs posterior height ratio
        c_min, c_max = int(occ_cols.min()), int(occ_cols.max())
        col_w = max(1, c_max - c_min)
        ant_cols = (occ_cols <= c_min + int(0.25 * col_w))
        post_cols = (occ_cols >= c_max - int(0.25 * col_w))
        mid_cols = (occ_cols > c_min + int(0.25 * col_w)) & (occ_cols < c_max - int(0.25 * col_w))

        h_ant = float(occ_rows[ant_cols].max() - occ_rows[ant_cols].min() + 1) if ant_cols.any() else h_px
        h_post = float(occ_rows[post_cols].max() - occ_rows[post_cols].min() + 1) if post_cols.any() else h_px
        h_mid = float(occ_rows[mid_cols].max() - occ_rows[mid_cols].min() + 1) if mid_cols.any() else h_px

        f_wedge_ratio = float(np.clip(h_ant / max(h_post, 1.0), 0.1, 2.0))
        f_mid_ratio = float(np.clip(h_mid / max(h_post, 1.0), 0.1, 2.0))
    else:
        f_aspect_ratio = 0.5
        f_wedge_ratio = 1.0
        f_mid_ratio = 1.0

    # 5. Endplate comminution (density variance across vertical thirds)
    if len(occ_rows) >= 10:
        r_min, r_max = int(occ_rows.min()), int(occ_rows.max())
        r_range = max(1, r_max - r_min)
        top_mask = body_mask & (np.arange(128)[:, None] <= r_min + int(0.33 * r_range))
        bot_mask = body_mask & (np.arange(128)[:, None] >= r_max - int(0.33 * r_range))
        top_val = float(np.mean(ch0_bone[top_mask])) if top_mask.any() else f_density_mean
        bot_val = float(np.mean(ch0_bone[bot_mask])) if bot_mask.any() else f_density_mean
        f_endplate_diff = float(np.abs(top_val - bot_val))
    else:
        f_endplate_diff = 0.0

    # 6. Posterior arch features (if present)
    if arch_roi is not None:
        arch_roi = np.asarray(arch_roi, dtype=np.float32)
        arch_bone = arch_roi[0]
        arch_soft = arch_roi[3]
        arch_mask = (arch_roi[4] >= 0.50)

        f_arch_density = float(np.mean(arch_bone[arch_mask])) if arch_mask.any() else 0.4
        f_interspinous_soft = float(np.mean(arch_soft))  # Edema / distraction in soft tissue
        f_arch_gradient = float(np.mean(np.abs(np.diff(arch_bone, axis=1))))
    else:
        f_arch_density = 0.4
        f_interspinous_soft = 0.2
        f_arch_gradient = 0.05

    # 7. Additional morphological & axial features
    f_body_area_norm = float(np.sum(body_mask) / (128.0 * 128.0))

    # Fix: Central 64x64 sub-region of axial slice body_roi[3, 32:96, 32:96]
    axial_center = ch3_axial[32:96, 32:96]
    f_axial_center_mean = float(np.mean(axial_center)) if axial_center.size > 0 else 0.3

    f_cortical_edge = float(np.mean(np.abs(np.diff(ch0_bone, axis=0))))

    base_features = [
        f_density_mean,        # 0: Bone density
        f_density_std,         # 1: Bone heterogeneity / trabecular comminution
        f_density_p90,         # 2: Cortical density peak
        f_para_asym,           # 3: Parasagittal fracture asymmetry
        f_canal_mean,          # 4: Retropulsed fragment density in canal
        f_canal_bone_ratio,    # 5: Canal stenosis area ratio (A3/A4)
        f_aspect_ratio,        # 6: Height/width ratio (vertebral collapse)
        f_wedge_ratio,         # 7: Ha/Hp ratio (A1 wedge vs normal)
        f_mid_ratio,           # 8: Hm/Hp ratio (fish-mouth or biconcave)
        f_endplate_diff,       # 9: Upper vs lower endplate comminution (A3 vs A4)
        f_arch_density,        # 10: Posterior arch bone density
        f_interspinous_soft,   # 11: Interspinous ligament edema (B2 tension band)
        f_arch_gradient,       # 12: Arch cortical disruption
        f_body_area_norm,      # 13: Vertebral body cross-sectional area
        f_axial_center_mean,   # 14: Central axial slice bone density (replaces faulty mask application)
        f_cortical_edge,       # 15: Overall cortical edge gradient
    ]

    # Optional: Concatenate geometric morphometry metrics if provided
    if morph_metrics is not None and isinstance(morph_metrics, dict):
        ha_hp = float(morph_metrics.get('ha_hp_ratio', 1.0))
        hm_hp = float(morph_metrics.get('hm_hp_ratio', 1.0))
        rel_collapse = float(morph_metrics.get('relative_collapse_ratio', min(ha_hp, hm_hp)))
        delta_post = float(morph_metrics.get('delta_posterior_mm', 0.0)) / 10.0  # scaled to ~[0, 1]
        isp = float(morph_metrics.get('isp_ratio', 1.0)) / 2.0                  # scaled to ~[0, 1]
        trans = float(morph_metrics.get('translation_mm', 0.0)) / 10.0          # scaled to ~[0, 1]
        base_features.extend([ha_hp, hm_hp, rel_collapse, delta_post, isp, trans])

    features = np.array(base_features, dtype=np.float32)

    # L2 normalize
    norm = np.linalg.norm(features)
    if norm > 1e-6:
        features /= norm

    return features


class AOSpinePrototypeBank:
    """2-Step Vertebra Prototype Bank with mathematical guarantee of Zero Patient Leakage.

    Architecture:
    - Step 1: `normal_prototypes_by_level` for T10..L5 + `global_normal_prototype`
      Computes Mahalanobis anomaly distance -> `p_anomaly` in [0.01, 0.99].
    - Step 2: `lesion_prototypes_by_class` for A1, A2, A3, A4, B2, B3, C
      Matches anomalous query to known lesion centroids with safe handling of missing classes.
    """

    def __init__(self, metric: str = 'cosine'):
        self.metric = metric
        self.fold_id: int | None = None
        self.feature_dim: int = 16

        # Z-score normalization fitted strictly on Train set
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None

        # Step 1: Normal prototypes by level
        self.normal_prototypes_by_level: dict[str, dict[str, Any]] = {}
        self.global_normal_prototype: dict[str, Any] = {}

        # Step 2: Lesion prototypes by class
        self.lesion_prototypes_by_class: dict[str, dict[str, Any]] = {}

        # Backward compatibility
        self.prototypes: dict[str, np.ndarray] = {}
        self.class_counts: dict[str, int] = {}

        # Leakage tracking
        self.trained_patient_ids: set[str] = set()
        self.trained_sample_ids: set[str] = set()
        self.external_ingested_count: int = 0

    def build_from_fold(
        self,
        samples: list[dict[str, Any]],
        train_sample_ids: list[str] | set[str],
        train_patient_ids: list[str] | set[str],
        roi_dir: Path | str | None = None,
        fold_id: int | None = None,
        use_morph_metrics: bool = False,
    ) -> None:
        """Build class prototypes and level-wise normal prototypes strictly from training samples of a fold.

        Parameters
        ----------
        samples : list of dict
            Full list of sample records with 'sample_id', 'patient_id', 'aospine_class', etc.
        train_sample_ids : list or set of str
            Allowed training sample IDs for this fold.
        train_patient_ids : list or set of str
            Allowed training patient IDs for this fold.
        roi_dir : Path or str, optional
            Base directory for ROI .npy files if relative paths are used.
        fold_id : int, optional
            Identifier of current fold.
        use_morph_metrics : bool, optional
            Whether to include morphometric metrics in feature extraction.
        """
        self.fold_id = fold_id
        train_sample_set = set(train_sample_ids)
        train_patient_set = set(train_patient_ids)

        self.prototypes.clear()
        self.class_counts.clear()
        self.normal_prototypes_by_level.clear()
        self.lesion_prototypes_by_class.clear()
        self.trained_patient_ids = set(train_patient_set)
        self.trained_sample_ids = set(train_sample_set)
        self.external_ingested_count = 0

        roi_root = Path(roi_dir) if roi_dir else None

        # 1. Collect raw training features
        train_records: list[dict[str, Any]] = []
        raw_feats_list: list[np.ndarray] = []

        for s in samples:
            sid = str(s.get('sample_id', ''))
            pid = str(s.get('patient_id', ''))
            cls_name = str(s.get('aospine_class', 'Normal'))
            lvl = str(s.get('level', '')).upper()

            if sid not in train_sample_set:
                continue

            # Strict assertion: patient must be an authorized training patient
            if pid not in train_patient_set:
                raise RuntimeError(
                    f"Zero Patient Leakage Violation: Sample {sid} belongs to patient {pid} "
                    f"which is NOT in the authorized training patient set!"
                )

            # Extract or retrieve features
            feats = s.get('prototype_features')
            if feats is None or use_morph_metrics:
                b_path = Path(s.get('body_roi_path', ''))
                if roi_root and not b_path.is_file():
                    b_path = roi_root / b_path.name
                if not b_path.is_file():
                    continue

                b_roi = np.load(b_path)
                a_roi = None
                a_path = Path(s.get('arch_roi_path', ''))
                if roi_root and not a_path.is_file():
                    a_path = roi_root / a_path.name
                if a_path.is_file():
                    a_roi = np.load(a_path)

                m_meta = s if use_morph_metrics else None
                feats = extract_roi_prototype_features(b_roi, a_roi, morph_metrics=m_meta)

            feats = np.asarray(feats, dtype=np.float32)
            raw_feats_list.append(feats)
            train_records.append({
                'sample_id': sid,
                'patient_id': pid,
                'level': lvl,
                'aospine_class': cls_name,
                'raw_features': feats,
            })

        if not raw_feats_list:
            raise RuntimeError("No training samples found matching train_sample_ids.")

        # 2. Compute fold-level Z-score statistics (mean & std) strictly from train set
        X_train = np.stack(raw_feats_list, axis=0)
        self.feature_dim = int(X_train.shape[1])
        self.feature_mean = np.mean(X_train, axis=0).astype(np.float32)
        std_calc = np.std(X_train, axis=0).astype(np.float32)
        self.feature_std = np.where(std_calc < 1e-6, 1.0, std_calc).astype(np.float32)

        # 3. Compute Z-scored representations for all training samples
        for r in train_records:
            z = (r['raw_features'] - self.feature_mean) / self.feature_std
            z_norm = z / (np.linalg.norm(z) + 1e-6)
            r['z_features'] = z
            r['z_norm_features'] = z_norm

        # 4. Step 1: Build normal prototypes by level (T10..L5) and global normal prototype
        normal_records = [r for r in train_records if r['aospine_class'] == 'Normal']
        if normal_records:
            all_norm_z = np.stack([r['z_features'] for r in normal_records], axis=0)
            g_std = np.std(all_norm_z, axis=0).astype(np.float32)
            g_std = np.maximum(g_std, 0.25).astype(np.float32)
            self.global_normal_prototype = {
                'mean': np.mean(all_norm_z, axis=0).astype(np.float32),
                'std': g_std,
                'count': len(normal_records),
            }
        else:
            self.global_normal_prototype = {
                'mean': np.zeros(self.feature_dim, dtype=np.float32),
                'std': np.ones(self.feature_dim, dtype=np.float32),
                'count': 0,
            }

        for lvl in SPINE_LEVELS:
            lvl_norm_recs = [r for r in normal_records if r['level'] == lvl]
            if len(lvl_norm_recs) >= 2:
                lvl_z_stack = np.stack([r['z_features'] for r in lvl_norm_recs], axis=0)
                lvl_mean = np.mean(lvl_z_stack, axis=0).astype(np.float32)
                lvl_std = np.std(lvl_z_stack, axis=0).astype(np.float32)
                lvl_std = np.maximum(lvl_std, 0.25).astype(np.float32)
                self.normal_prototypes_by_level[lvl] = {
                    'mean': lvl_mean,
                    'std': lvl_std,
                    'count': len(lvl_norm_recs),
                }
            else:
                # Fallback to global normal prototype
                self.normal_prototypes_by_level[lvl] = {
                    'mean': self.global_normal_prototype['mean'].copy(),
                    'std': self.global_normal_prototype['std'].copy(),
                    'count': len(lvl_norm_recs),
                }

        # 5. Step 2: Build lesion prototypes by class (A1..C)
        for c in ALL_AOSPINE_CLASSES:
            c_recs = [r for r in train_records if r['aospine_class'] == c]
            if c_recs:
                stacked_z = np.stack([r['z_norm_features'] for r in c_recs], axis=0)
                centroid = np.mean(stacked_z, axis=0).astype(np.float32)
                c_norm = np.linalg.norm(centroid)
                if c_norm > 1e-6:
                    centroid /= c_norm
                std_c = np.std(stacked_z, axis=0).astype(np.float32)
                self.lesion_prototypes_by_class[c] = {
                    'has_samples': True,
                    'sample_count': len(c_recs),
                    'centroid': centroid,
                    'std': std_c,
                }
                self.prototypes[c] = centroid
                self.class_counts[c] = len(c_recs)
            else:
                self.lesion_prototypes_by_class[c] = {
                    'has_samples': False,
                    'sample_count': 0,
                    'centroid': None,
                    'std': None,
                }

        # Also store Normal in prototypes dict for backward compatibility
        norm_centroid = self.global_normal_prototype['mean'].copy()
        n_len = np.linalg.norm(norm_centroid)
        if n_len > 1e-6:
            norm_centroid /= n_len
        self.prototypes['Normal'] = norm_centroid
        self.class_counts['Normal'] = len(normal_records)

    def ingest_external_normals(self, external_dir: Path | str, external_weight: float = 0.5) -> int:
        """Ingest external multi-scanner clean normal vertebrae to update normal distribution (mean & std).

        Applies weighted pooled variance:
            V_pooled = (1 - w) * V_local + w * V_ext + (1 - w) * w * (mu_local - mu_ext)^2
            std_pooled = max(sqrt(V_pooled), 0.25)

        Parameters
        ----------
        external_dir : Path or str
            Directory containing external normal manifest.json or pre-extracted ROIs.
        external_weight : float
            Weight w in (0, 1) assigned to external normal distribution (default: 0.5).

        Returns
        -------
        int
            Number of successfully ingested normal vertebrae.
        """
        if self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("Prototype bank must be built or loaded before ingesting external normals.")

        ext_d = Path(external_dir)
        if not ext_d.exists():
            return 0

        w = float(np.clip(external_weight, 0.01, 0.99))
        w1 = 1.0 - w
        w2 = w

        manifest_file = ext_d / "manifest.json"
        ext_records_by_level: dict[str, list[np.ndarray]] = {lvl: [] for lvl in SPINE_LEVELS}
        all_ext_z: list[np.ndarray] = []

        if manifest_file.is_file():
            try:
                m_data = json.loads(manifest_file.read_text(encoding='utf-8'))
                for s in m_data.get('samples', []):
                    lvl = str(s.get('level', '')).upper()
                    if lvl not in SPINE_LEVELS:
                        continue
                    feats = s.get('prototype_features')
                    if feats is not None and len(feats) == self.feature_dim:
                        f_arr = np.array(feats, dtype=np.float32)
                    elif s.get('body_roi_path'):
                        b_p = ext_d / s['body_roi_path']
                        a_p = ext_d / s.get('arch_roi_path', '')
                        if not b_p.is_file():
                            continue
                        b_roi = np.load(b_p)
                        a_roi = np.load(a_p) if a_p.is_file() else None
                        f_arr = extract_roi_prototype_features(b_roi, a_roi)
                    else:
                        continue

                    if np.isnan(f_arr).any() or np.isinf(f_arr).any():
                        continue

                    z = (f_arr - self.feature_mean) / self.feature_std
                    ext_records_by_level[lvl].append(z)
                    all_ext_z.append(z)
            except Exception as e:
                print(f"[PrototypeBank] Warning loading external manifest {manifest_file}: {e}")

        # If manifest not present or yielded 0 records, try scanning *_body.npy files
        if not all_ext_z:
            roi_files = list(ext_d.glob('*_body.npy'))
            for b_file in roi_files:
                try:
                    b_roi = np.load(b_file)
                    a_file = b_file.with_name(b_file.name.replace('_body.npy', '_arch.npy'))
                    a_roi = np.load(a_file) if a_file.is_file() else None
                    feats = extract_roi_prototype_features(b_roi, a_roi)
                    if np.isnan(feats).any() or np.isinf(feats).any():
                        continue

                    z = (feats - self.feature_mean) / self.feature_std
                    lvl_candidate = 'T12'
                    for l_name in SPINE_LEVELS:
                        if f"_{l_name}_" in b_file.name or b_file.name.endswith(f"_{l_name}.npy"):
                            lvl_candidate = l_name
                            break
                    ext_records_by_level[lvl_candidate].append(z)
                    all_ext_z.append(z)
                except Exception:
                    continue

        if not all_ext_z:
            return 0

        # Update level-wise normal prototypes
        for lvl in SPINE_LEVELS:
            lvl_records = ext_records_by_level[lvl]
            if not lvl_records:
                continue

            ext_stack = np.stack(lvl_records, axis=0)
            mu_ext = np.mean(ext_stack, axis=0).astype(np.float32)
            v_ext = np.var(ext_stack, axis=0).astype(np.float32) if len(lvl_records) >= 2 else np.full(self.feature_dim, 0.25**2, dtype=np.float32)

            curr_proto = self.normal_prototypes_by_level.get(lvl)
            if curr_proto is not None and curr_proto.get('mean') is not None and curr_proto.get('count', 0) > 0:
                mu_local = curr_proto['mean']
                v_local = (curr_proto['std'] ** 2).astype(np.float32)
                old_cnt = curr_proto['count']
            else:
                mu_local = self.global_normal_prototype.get('mean', np.zeros(self.feature_dim, dtype=np.float32))
                v_local = (self.global_normal_prototype.get('std', np.ones(self.feature_dim, dtype=np.float32)) ** 2).astype(np.float32)
                old_cnt = 0

            # Weighted pooled mean & variance
            mu_pooled = (w1 * mu_local + w2 * mu_ext).astype(np.float32)
            v_pooled = (w1 * v_local + w2 * v_ext + w1 * w2 * ((mu_local - mu_ext) ** 2)).astype(np.float32)
            sigma_pooled = np.maximum(np.sqrt(np.maximum(v_pooled, 1e-6)), 0.25).astype(np.float32)

            self.normal_prototypes_by_level[lvl] = {
                'mean': mu_pooled,
                'std': sigma_pooled,
                'count': old_cnt + len(lvl_records),
            }

        # Update global normal prototype
        all_ext_stack = np.stack(all_ext_z, axis=0)
        mu_ext_g = np.mean(all_ext_stack, axis=0).astype(np.float32)
        v_ext_g = np.var(all_ext_stack, axis=0).astype(np.float32)

        mu_local_g = self.global_normal_prototype.get('mean', np.zeros(self.feature_dim, dtype=np.float32))
        v_local_g = (self.global_normal_prototype.get('std', np.ones(self.feature_dim, dtype=np.float32)) ** 2).astype(np.float32)
        old_cnt_g = self.global_normal_prototype.get('count', 0)

        mu_pooled_g = (w1 * mu_local_g + w2 * mu_ext_g).astype(np.float32)
        v_pooled_g = (w1 * v_local_g + w2 * v_ext_g + w1 * w2 * ((mu_local_g - mu_ext_g) ** 2)).astype(np.float32)
        sigma_pooled_g = np.maximum(np.sqrt(np.maximum(v_pooled_g, 1e-6)), 0.25).astype(np.float32)

        self.global_normal_prototype = {
            'mean': mu_pooled_g,
            'std': sigma_pooled_g,
            'count': old_cnt_g + len(all_ext_z),
        }

        # Update backward compatibility Normal prototype
        norm_centroid = mu_pooled_g.copy()
        n_len = np.linalg.norm(norm_centroid)
        if n_len > 1e-6:
            norm_centroid /= n_len
        self.prototypes['Normal'] = norm_centroid
        self.class_counts['Normal'] = self.global_normal_prototype['count']

        self.external_ingested_count += len(all_ext_z)
        return len(all_ext_z)

    def ingest_verse_normals(self, verse_dir: Path | str) -> int:
        """Alias for ingest_external_normals for backward compatibility."""
        return self.ingest_external_normals(verse_dir)

    def compute_normal_anomaly_score(
        self,
        query_features: np.ndarray,
        level: str | None = None,
        d_threshold: float = 1.50,
        slope: float = 2.0,
    ) -> dict[str, Any]:
        """Step 1: Compute anomaly score against normal vertebra prototype for the specified level.

        Parameters
        ----------
        query_features : np.ndarray
            Feature vector of shape (D,).
        level : str, optional
            Spine level (e.g. 'T12', 'L1', 'L4'). If None or missing, falls back to global normal prototype.
        d_threshold : float
            Distance inflection point for sigmoid calibration (default 1.50).
        slope : float
            Sigmoid steepness factor (default 2.0).

        Returns
        -------
        dict
            Result with 'p_anomaly' in [0.01, 0.99], 'anomaly_distance', 'is_anomaly', 'reference_level', 'level_matched'.
        """
        if self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("Prototype bank is not initialized. Call build_from_fold or load first.")

        q = np.asarray(query_features, dtype=np.float32)
        z = (q - self.feature_mean) / self.feature_std

        lvl_clean = str(level).strip().upper() if level else ''
        level_matched = False
        if lvl_clean in self.normal_prototypes_by_level and self.normal_prototypes_by_level[lvl_clean]['count'] >= 2:
            ref_proto = self.normal_prototypes_by_level[lvl_clean]
            ref_level = lvl_clean
            level_matched = True
        else:
            ref_proto = self.global_normal_prototype
            ref_level = 'global'

        mu = ref_proto['mean']
        sigma = ref_proto['std']

        # Mahalanobis-diagonal normalized distance
        diff = (z - mu) / sigma
        d = float(np.sqrt(np.mean(diff ** 2)))

        # Sigmoid mapping to p_anomaly in [0.01, 0.99]
        logit = slope * (d - d_threshold)
        p_anom = float(1.0 / (1.0 + np.exp(-np.clip(logit, -10.0, 10.0))))
        p_anom = float(np.clip(p_anom, 0.01, 0.99))

        return {
            'p_anomaly': float(round(p_anom, 4)),
            'anomaly_distance': float(round(d, 4)),
            'is_anomaly': bool(p_anom >= 0.50),
            'reference_level': ref_level,
            'level_matched': bool(level_matched),
        }

    def match_lesion_prototypes(
        self,
        query_features: np.ndarray,
        temperature: float = 5.0,
    ) -> dict[str, Any]:
        """Step 2: Match an anomalous vertebra query against available lesion prototypes.

        Parameters
        ----------
        query_features : np.ndarray
            Feature vector of shape (D,).
        temperature : float
            Softmax scaling factor for confidence calculation.

        Returns
        -------
        dict
            Result with 'predicted_lesion_class', 'confidence', 'similarities', 'distances', 'confidence_distribution'.
        """
        if self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("Prototype bank is not initialized. Call build_from_fold or load first.")

        q = np.asarray(query_features, dtype=np.float32)
        z = (q - self.feature_mean) / self.feature_std
        z_norm = z / (np.linalg.norm(z) + 1e-6)

        similarities: dict[str, float] = {}
        distances: dict[str, float] = {}

        for c, proto in self.lesion_prototypes_by_class.items():
            if not proto['has_samples'] or proto['centroid'] is None:
                continue
            cos_sim = float(np.dot(z_norm, proto['centroid']))
            euc_dist = float(np.linalg.norm(z_norm - proto['centroid']))
            similarities[c] = float(np.clip(cos_sim, -1.0, 1.0))
            distances[c] = euc_dist

        if not similarities:
            # Fallback if no lesion classes had training samples
            return {
                'predicted_lesion_class': 'A3',
                'confidence': 0.50,
                'similarities': {'A3': 0.0},
                'distances': {'A3': 1.0},
                'confidence_distribution': {'A3': 1.0},
                'available_classes': [],
            }

        best_cls = max(similarities.keys(), key=lambda c: similarities[c])

        # Softmax confidence over available classes
        sim_vals = np.array(list(similarities.values()))
        exp_sim = np.exp((sim_vals - np.max(sim_vals)) * temperature)
        conf_vals = exp_sim / np.sum(exp_sim)
        conf_dict = {c: float(conf_vals[i]) for i, c in enumerate(similarities.keys())}

        return {
            'predicted_lesion_class': best_cls,
            'confidence': float(conf_dict[best_cls]),
            'similarities': similarities,
            'distances': distances,
            'confidence_distribution': conf_dict,
            'available_classes': list(similarities.keys()),
        }

    def predict(
        self,
        query_features: np.ndarray,
        level: str | None = None,
        anomaly_threshold: float = 0.50,
    ) -> dict[str, Any]:
        """Perform unified 2-step inference (Step 1 Anomaly Detection -> Step 2 Lesion Classification).

        Maintains 100% backward compatibility with previous predict() signature and return keys.

        Parameters
        ----------
        query_features : np.ndarray
            Feature vector of shape (D,).
        level : str, optional
            Vertebral level (e.g. 'T12', 'L1') for level-specific normal reference.
        anomaly_threshold : float
            Threshold on p_anomaly to decide Normal vs Fractured (default 0.50).

        Returns
        -------
        dict
            Prediction dictionary with 'predicted_class', 'confidence', 'similarities',
            'distances', 'confidence_distribution', 'p_anomaly', 'anomaly_distance', 'is_anomaly'.
        """
        step1 = self.compute_normal_anomaly_score(query_features, level=level)
        step2 = self.match_lesion_prototypes(query_features)

        p_anom = step1['p_anomaly']

        all_sims = dict(step2['similarities'])
        all_dists = dict(step2['distances'])
        # Normal similarity inversely related to anomaly distance
        norm_sim = float(np.clip(1.0 - (step1['anomaly_distance'] / 3.0), -1.0, 1.0))
        all_sims['Normal'] = norm_sim
        all_dists['Normal'] = float(step1['anomaly_distance'])

        if p_anom < anomaly_threshold:
            pred_cls = 'Normal'
            conf = float(round(1.0 - p_anom, 4))
        else:
            pred_cls = step2['predicted_lesion_class']
            conf = float(round(step2['confidence'], 4))

        # Softmax over all classes including Normal
        sim_items = list(all_sims.items())
        vals = np.array([v for _, v in sim_items])
        exp_v = np.exp((vals - np.max(vals)) * 3.0)
        c_vals = exp_v / np.sum(exp_v)
        full_conf = {k: float(round(c_vals[i], 4)) for i, (k, _) in enumerate(sim_items)}

        return {
            'predicted_class': pred_cls,
            'confidence': conf,
            'similarities': all_sims,
            'distances': all_dists,
            'confidence_distribution': full_conf,
            'p_anomaly': p_anom,
            'anomaly_distance': step1['anomaly_distance'],
            'is_anomaly': step1['is_anomaly'],
            'reference_level': step1['reference_level'],
        }

    def evaluate_fold_val(
        self,
        samples: list[dict[str, Any]],
        val_sample_ids: list[str] | set[str],
        val_patient_ids: list[str] | set[str],
        roi_dir: Path | str | None = None,
        use_2step: bool = True,
        anomaly_threshold: float = 0.50,
    ) -> dict[str, Any]:
        """Evaluate validation set with mathematical verification of Zero Patient Leakage.

        Parameters
        ----------
        samples : list of dict
            Full list of sample records.
        val_sample_ids : list or set of str
            Allowed validation sample IDs for this fold.
        val_patient_ids : list or set of str
            Validation patient IDs for this fold.
        roi_dir : Path or str, optional
            Base directory for ROI .npy files.
        use_2step : bool
            Whether to use 2-step architecture (True) or legacy 1-step (False).
        anomaly_threshold : float
            Threshold on p_anomaly for 2-step decision.

        Returns
        -------
        dict
            Evaluation summary with 'accuracy', 'sample_count', 'leakage_verified', 'predictions'.
        """
        val_sample_set = set(val_sample_ids)
        val_patient_set = set(val_patient_ids)

        # 1. Zero Patient Leakage Verification
        leakage = self.trained_patient_ids & val_patient_set
        if leakage:
            raise RuntimeError(
                f"FATAL: Patient Leakage detected between Prototype Bank and Validation Set! "
                f"Leaked patients: {leakage}"
            )

        roi_root = Path(roi_dir) if roi_dir else None
        correct = 0
        total = 0
        predictions: list[dict[str, Any]] = []

        for s in samples:
            sid = str(s.get('sample_id', ''))
            pid = str(s.get('patient_id', ''))
            gt_cls = str(s.get('aospine_class', 'Normal'))
            lvl = str(s.get('level', '')).upper()

            if sid not in val_sample_set:
                continue

            feats = s.get('prototype_features')
            if feats is None:
                b_path = Path(s.get('body_roi_path', ''))
                if roi_root and not b_path.is_file():
                    b_path = roi_root / b_path.name
                if not b_path.is_file():
                    continue

                b_roi = np.load(b_path)
                a_roi = None
                a_path = Path(s.get('arch_roi_path', ''))
                if roi_root and not a_path.is_file():
                    a_path = roi_root / a_path.name
                if a_path.is_file():
                    a_roi = np.load(a_path)

                feats = extract_roi_prototype_features(b_roi, a_roi)

            pred = self.predict(feats, level=lvl, anomaly_threshold=anomaly_threshold)
            is_match = (pred['predicted_class'] == gt_cls)
            if is_match:
                correct += 1
            total += 1

            predictions.append({
                'sample_id': sid,
                'patient_id': pid,
                'level': lvl,
                'true_class': gt_cls,
                'predicted_class': pred['predicted_class'],
                'confidence': pred['confidence'],
                'p_anomaly': pred['p_anomaly'],
                'anomaly_distance': pred['anomaly_distance'],
                'is_correct': is_match,
            })

        acc = float(correct / total) if total > 0 else 0.0
        return {
            'fold_id': self.fold_id,
            'accuracy': acc,
            'total_val_samples': total,
            'correct_count': correct,
            'zero_leakage_verified': True,
            'trained_patients_count': len(self.trained_patient_ids),
            'val_patients_count': len(val_patient_set),
            'prototypes_available': [c for c, p in self.lesion_prototypes_by_class.items() if p['has_samples']],
            'predictions': predictions,
        }

    def save(self, filepath: Path | str) -> None:
        """Serialize full prototype bank state to a JSON file."""
        f_path = Path(filepath)
        f_path.parent.mkdir(parents=True, exist_ok=True)

        normal_levels_ser = {}
        for lvl, data in self.normal_prototypes_by_level.items():
            normal_levels_ser[lvl] = {
                'mean': data['mean'].tolist() if data['mean'] is not None else None,
                'std': data['std'].tolist() if data['std'] is not None else None,
                'count': int(data['count']),
            }

        lesion_classes_ser = {}
        for c, data in self.lesion_prototypes_by_class.items():
            lesion_classes_ser[c] = {
                'has_samples': bool(data['has_samples']),
                'sample_count': int(data['sample_count']),
                'centroid': data['centroid'].tolist() if data['centroid'] is not None else None,
                'std': data['std'].tolist() if data['std'] is not None else None,
            }

        protos_ser = {c: p.tolist() for c, p in self.prototypes.items()}

        payload = {
            'schema_version': 2,
            'fold_id': self.fold_id,
            'feature_dim': self.feature_dim,
            'metric': self.metric,
            'feature_mean': self.feature_mean.tolist() if self.feature_mean is not None else None,
            'feature_std': self.feature_std.tolist() if self.feature_std is not None else None,
            'normal_prototypes_by_level': normal_levels_ser,
            'global_normal_prototype': {
                'mean': self.global_normal_prototype['mean'].tolist() if self.global_normal_prototype.get('mean') is not None else None,
                'std': self.global_normal_prototype['std'].tolist() if self.global_normal_prototype.get('std') is not None else None,
                'count': int(self.global_normal_prototype.get('count', 0)),
            },
            'lesion_prototypes_by_class': lesion_classes_ser,
            'prototypes': protos_ser,
            'class_counts': self.class_counts,
            'trained_patient_ids': sorted(list(self.trained_patient_ids)),
            'trained_sample_ids': sorted(list(self.trained_sample_ids)),
            'external_ingested_count': int(self.external_ingested_count),
        }

        f_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    def load(self, filepath: Path | str) -> 'AOSpinePrototypeBank':
        """Load prototype bank state from a JSON file."""
        f_path = Path(filepath)
        data = json.loads(f_path.read_text(encoding='utf-8'))

        self.fold_id = data.get('fold_id')
        self.feature_dim = int(data.get('feature_dim', 16))
        self.metric = data.get('metric', 'cosine')

        if data.get('feature_mean') is not None:
            self.feature_mean = np.array(data['feature_mean'], dtype=np.float32)
        if data.get('feature_std') is not None:
            self.feature_std = np.array(data['feature_std'], dtype=np.float32)

        self.normal_prototypes_by_level.clear()
        for lvl, p_data in data.get('normal_prototypes_by_level', {}).items():
            self.normal_prototypes_by_level[lvl] = {
                'mean': np.array(p_data['mean'], dtype=np.float32) if p_data.get('mean') is not None else None,
                'std': np.array(p_data['std'], dtype=np.float32) if p_data.get('std') is not None else None,
                'count': int(p_data.get('count', 0)),
            }

        g_data = data.get('global_normal_prototype', {})
        self.global_normal_prototype = {
            'mean': np.array(g_data['mean'], dtype=np.float32) if g_data.get('mean') is not None else None,
            'std': np.array(g_data['std'], dtype=np.float32) if g_data.get('std') is not None else None,
            'count': int(g_data.get('count', 0)),
        }

        self.lesion_prototypes_by_class.clear()
        for c, l_data in data.get('lesion_prototypes_by_class', {}).items():
            self.lesion_prototypes_by_class[c] = {
                'has_samples': bool(l_data.get('has_samples', False)),
                'sample_count': int(l_data.get('sample_count', 0)),
                'centroid': np.array(l_data['centroid'], dtype=np.float32) if l_data.get('centroid') is not None else None,
                'std': np.array(l_data['std'], dtype=np.float32) if l_data.get('std') is not None else None,
            }

        self.prototypes.clear()
        for c, p_vec in data.get('prototypes', {}).items():
            self.prototypes[c] = np.array(p_vec, dtype=np.float32)

        self.class_counts = dict(data.get('class_counts', {}))
        self.trained_patient_ids = set(data.get('trained_patient_ids', []))
        self.trained_sample_ids = set(data.get('trained_sample_ids', []))
        self.external_ingested_count = int(data.get('external_ingested_count', 0))
        return self
