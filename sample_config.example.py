"""Copy to sample_config.py, change SAMPLE_DIR, then restart the demo."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / 'data/raw/sample_patient'
