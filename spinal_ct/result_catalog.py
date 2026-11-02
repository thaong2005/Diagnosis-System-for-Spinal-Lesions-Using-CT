"""Find local results by CT identity; opening still validates their geometry."""
import json
from pathlib import Path


def discover_results(roots, fingerprint):
    if isinstance(roots, (str, Path)):
        roots = [roots]
    matches, unreadable, seen = [], 0, set()
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for path in root_path.glob('**/prediction.json'):
            if any(part.startswith('.') for part in path.relative_to(root_path).parts):
                continue
            folder = path.parent.resolve()
            if folder in seen:
                continue
            try:
                report = json.loads(path.read_text(encoding='utf-8'))
                if not isinstance(report, dict):
                    continue
                if (report.get('task') != 'vertebral_localization'
                        or report.get('input_fingerprint') != fingerprint):
                    continue
                required = [path.parent/'vertebrae.nii.gz']
                if 'monte_carlo' in report:
                    required += [path.parent/'monte_carlo'/name for name in ('consensus.nii.gz', 'disagreement.nii.gz')]
                if all(p.is_file() for p in required):
                    seen.add(folder)
                    matches.append(dict(folder=path.parent, report=report))
            except (OSError, ValueError, TypeError):
                unreadable += 1
    matches.sort(key=lambda item: str(item['report'].get('created_at_utc', '')), reverse=True)
    return matches, unreadable


def compatible_result(report, *, fast, samples=0, noise=20., seed=42):
    model = report.get('model') or {}
    if not isinstance(model, dict):
        return False
    if (model.get('backend') != 'TotalSegmentator' or model.get('version') != '2.18.0'
            or model.get('resolution') != ('3mm' if fast else '1.5mm')
            or model.get('device') != 'cpu' or model.get('robust_crop') != (not fast)):
        return False
    mc = report.get('monte_carlo') or {}
    if not isinstance(mc, dict):
        return False
    if not samples:
        return not mc
    return (mc.get('samples') == samples and mc.get('noise_sigma_hu') == noise
            and mc.get('seed') == seed and mc.get('method') == 'monte_carlo_input_perturbation')


def discover_regions(roots, fingerprint):
    """Latest saved draft per source/level; full integrity is checked on opening."""
    if isinstance(roots, (str, Path)):
        roots = [roots]
    regions, seen = {}, set()
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for path in root_path.glob('**/region.json'):
            if any(part.startswith('.') for part in path.relative_to(root_path).parts):
                continue
            folder = path.parent.resolve()
            if folder in seen:
                continue
            seen.add(folder)
            try:
                document = json.loads(path.read_text(encoding='utf-8'))
                if (not isinstance(document, dict) or document.get('task') != 'vertebral_body_region'
                        or document.get('input_fingerprint') != fingerprint
                        or not (path.parent/'body_mask.npy').is_file()):
                    continue
                key = (document.get('source_kind'), document.get('level'))
                created = str(document.get('created_utc', ''))
                if key not in regions or created > regions[key][0]:
                    regions[key] = (created, path.parent)
            except (OSError, ValueError, TypeError):
                continue
    return {key: item[1] for key, item in regions.items()}
