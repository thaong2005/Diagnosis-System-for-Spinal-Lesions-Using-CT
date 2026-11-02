"""Background computations; delivery always happens on the Qt GUI thread."""
import time
from PySide6.QtCore import QThread


class Task(QThread):
    def __init__(self, function, parent=None):
        super().__init__(parent)
        self.function = function
        self.value = None
        self.error = None
        self.elapsed = 0.

    def run(self):
        started = time.perf_counter()
        try:
            self.value = self.function()
        except Exception as error:
            self.error = str(error)
        finally:
            self.elapsed = time.perf_counter() - started
