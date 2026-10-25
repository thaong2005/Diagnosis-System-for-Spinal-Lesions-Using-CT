"""Load saved results for physical-space review and export a local HTML report."""
from __future__ import annotations

import hashlib
import html
import json
from itertools import product
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def ct_fingerprint(image: sitk.Image) -> str:
    """Hash HU voxels and geometry, independent of patient metadata and file format."""
    geometry = [image.GetSize(), image.GetSpacing(), image.GetOrigin(), image.GetDirection()]
    digest = hashlib.sha256(json.dumps(geometry).encode('ascii'))
    for plane in sitk.GetArrayViewFromImage(image):
        digest.update(np.asarray(plane, dtype='<f4').tobytes())
    return digest.hexdigest()


def same_geometry(a: sitk.Image, b: sitk.Image) -> bool:
    return (a.GetDimension() == b.GetDimension() == 3
            and a.GetNumberOfComponentsPerPixel() == b.GetNumberOfComponentsPerPixel() == 1
            and a.GetSize() == b.GetSize()
            and np.allclose(a.GetSpacing(), b.GetSpacing(), rtol=0, atol=1e-5)
            and np.allclose(a.GetOrigin(), b.GetOrigin(), rtol=0, atol=1e-4)
            and np.allclose(a.GetDirection(), b.GetDirection(), rtol=0, atol=1e-5))


def display_image(image: sitk.Image) -> sitk.Image:
    """Axis-aligned LPS display grid. Original CT/masks are never modified."""
    oriented = sitk.DICOMOrient(image, 'LPS')
    if np.allclose(oriented.GetDirection(), np.eye(3).ravel(), atol=1e-6):
        return oriented
    corners = np.array([image.TransformContinuousIndexToPhysicalPoint(tuple(map(float, p)))
                        for p in product(*[(-.5, n-.5) for n in image.GetSize()])])
    spacing = np.array(oriented.GetSpacing())
    size = np.ceil(np.ptp(corners, axis=0)/spacing).astype(int)
    if np.prod(size.astype(float)) > 300_000_000:
        raise ValueError('Lưới hiển thị ảnh xiên quá lớn; cần giảm kích thước ca trước.')
    reference = sitk.Image([int(n) for n in size], sitk.sitkFloat32)
    reference.SetOrigin((corners.min(axis=0)+spacing/2).tolist())
    reference.SetSpacing(spacing.tolist())
    return sitk.Resample(image, reference, sitk.Transform(), sitk.sitkLinear, -1024)


def load_review(folder: Path, source: sitk.Image, reference: sitk.Image,
                fingerprint: str) -> dict:
    report = json.loads((folder/'prediction.json').read_text(encoding='utf-8'))
    if not isinstance(report, dict) or report.get('task') != 'vertebral_localization':
        raise ValueError('Thư mục không chứa báo cáo định vị hợp lệ.')
    verified = report.get('input_fingerprint') == fingerprint
    if report.get('input_fingerprint') and not verified:
        raise ValueError('Kết quả thuộc CT khác. Hãy chọn đúng ảnh gốc.')
    warnings = list(report.get('warnings', []))
    if not verified:
        warnings.insert(0, 'Kết quả cũ chưa có mã kiểm tra CT: chỉ khớp geometry; cần xác nhận đúng ca.')
    masks = {}
    paths = {'Baseline': folder/'vertebrae.nii.gz'}
    if 'monte_carlo' in report:
        paths.update({'Consensus': folder/'monte_carlo/consensus.nii.gz',
                      'Bất đồng': folder/'monte_carlo/disagreement.nii.gz'})
    for name, path in paths.items():
        mask = sitk.ReadImage(str(path))
        if not same_geometry(source, mask):
            raise ValueError(f'{name}: geometry khác ảnh CT đang mở.')
        values = sitk.GetArrayViewFromImage(mask)
        if not np.isfinite(values).all():
            raise ValueError(f'{name}: mask chứa giá trị không hữu hạn.')
        if name == 'Bất đồng':
            if values.min() < 0 or values.max() > 1:
                raise ValueError('Bản đồ bất đồng phải trong khoảng 0–1.')
        elif not np.isin(values, np.arange(26)).all():
            raise ValueError('Mask chứa mã đốt sống không hợp lệ.')
        aligned = sitk.Resample(mask, reference, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
        masks[name] = sitk.GetArrayFromImage(aligned)
    return dict(report=report, masks=masks, warnings=warnings, verified=verified, folder=folder)


def report_html(report: dict, warnings: list[str], case_alias: str) -> str:
    """Standalone text report: includes no source paths or patient metadata."""
    esc = lambda value: html.escape(str(value))
    rows = ''.join('<tr><td>'+esc(p['level'])+'</td><td>'+f"{p['volume_mm3']:,.1f}"+
                   '</td><td>'+('Chạm biên' if p.get('touches_boundary') else '—')+'</td></tr>'
                   for p in report.get('predictions', []))
    model = report.get('model', {})
    mc = report.get('monte_carlo', {})
    mc_rows = ''.join(f"<tr><td>{esc(p['level'])}</td><td>{p['detected_samples']}/{mc.get('samples', '?')}</td>"
                      f"<td>{p['volume_mean_mm3']:,.1f}</td><td>{p['volume_std_mm3']:,.1f}</td></tr>"
                      for p in mc.get('per_level', []) if p['detected_samples'])
    return f'''<!doctype html><html lang="vi"><meta charset="utf-8">
<title>Báo cáo CT — {esc(case_alias)}</title>
<style>body{{font:16px system-ui;max-width:960px;margin:40px auto;padding:0 24px;color:#203344}}
table{{border-collapse:collapse;width:100%;margin:20px 0}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #ddd}}
h1,h2{{color:#125a67}}li{{margin:8px 0}}@media print{{body{{margin:0}}}}</style>
<h1>Định vị đốt sống · {esc(case_alias)}</h1>
<p>Trạng thái: <b>{esc(report.get('status'))}</b> · Cần kiểm tra kết quả.</p>
<p>Backend: {esc(model.get('backend'))} {esc(model.get('version'))} ·
Độ phân giải: {esc(model.get('resolution'))} · Thiết bị: {esc(model.get('device'))}</p>
<p>Thời điểm UTC: {esc(report.get('created_at_utc', 'Không ghi nhận'))} ·
Thời gian chạy (giây): {esc(report.get('runtime_seconds', 'Không ghi nhận'))}</p>
<h2>Baseline</h2><table><tr><th>Đốt</th><th>Thể tích toàn đốt (mm³)</th><th>Cảnh báo</th></tr>{rows}</table>
<h2>Monte Carlo</h2><p>Số mẫu: {esc(mc.get('samples', 0))} · Sigma HU: {esc(mc.get('noise_sigma_hu', '—'))}
· Seed: {esc(mc.get('seed', '—'))}</p>
<table><tr><th>Đốt</th><th>Phát hiện</th><th>Thể tích TB (mm³)</th><th>Độ lệch chuẩn (mm³)</th></tr>{mc_rows}</table>
<h2>Lưu ý của lần chạy</h2><ul>{''.join('<li>'+esc(w)+'</li>' for w in warnings)}</ul>
<p>Thể tích mô tả mask toàn đốt sống. Chưa tính diện tích hoặc phân loại AO.
Đồng thuận với nhiễu HU không phải độ chính xác hay xác suất bệnh.
Đánh giá đúng/sai cần chuẩn tham chiếu đã được chuyên gia duyệt.</p></html>'''
