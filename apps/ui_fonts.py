"""Font fallback shared by the demo and standalone offscreen review dialogs."""
from pathlib import Path
from matplotlib import get_data_path
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication


def ui_font():
    for path in (Path('C:/Windows/Fonts/segoeui.ttf'), Path(get_data_path())/'fonts/ttf/DejaVuSans.ttf'):
        if path.is_file():
            families = QFontDatabase.applicationFontFamilies(QFontDatabase.addApplicationFont(str(path)))
            if families:
                return families[0]
    return QApplication.font().family()
