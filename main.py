from vgg.nst_container import NST
from views.Open import Open
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys


def main():
    app = QApplication(sys.argv)
    nst = NST()
    #nst.STYLE_PATH = "C:\\Users\\bende\\Pictures\\monet.jpg"
    #nst.CONTENT_PATH = "C:\\Users\\bende\\Pictures\\smugTrump.png"
    #res = nst.neural_style_transfer(nst.CONTENT_PATH, nst.STYLE_PATH)
    main_window = QMainWindow()
    open_window = Open(nst)
    open_window.setupUi(main_window)
    main_window.show()
    app.exec()
    return


if __name__ == "__main__":
    main()
