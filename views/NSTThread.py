from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal, QObject
import time

class NSTThread(QThread, QObject):

    image_update = pyqtSignal()
    progress_update = pyqtSignal(float)


    def __init__(self, nst_containter, options):
        self.nst = nst_containter
        self.options = options
        QThread.__init__(self)
        return

    def __del__(self):
        self.wait()
        return

    def run(self):
        counter = 0
        INITIALIZE_TIMEOUT = 60
        INITIALIZE_SLEEP = 2
        TRAINING_TIMEOUT = 60 * 20
        TRAINING_SLEEP = 4
        while self.nst.EPOCH_COMPLETE < 1 and counter < INITIALIZE_TIMEOUT:
            counter+=INITIALIZE_SLEEP
            time.sleep(INITIALIZE_SLEEP)
        if counter >= INITIALIZE_TIMEOUT:
            QMessageBox.information(self, "Timeout", "Timeout")
            return
        counter = 0
        while self.nst.EPOCH_COMPLETE < self.nst.MAX_EPOCH - 1 and counter < TRAINING_TIMEOUT:
            counter += TRAINING_SLEEP
            self.image_update.emit()
            self.progress_update.emit((self.nst.EPOCH_COMPLETE/self.nst.MAX_EPOCH) * 100)
            self.sleep(TRAINING_SLEEP)
        return