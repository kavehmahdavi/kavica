import sys
from PyQt5.QtWidgets import (QApplication, QCheckBox, QGridLayout, QGroupBox,
                             QMenu, QPushButton, QRadioButton, QVBoxLayout, QWidget)
from PyQt5.QtWidgets import QMainWindow, QFrame, QApplication, QWidget, QAction, QTextEdit, QLabel, QVBoxLayout
from PyQt5.QtCore import pyqtSlot
from functools import partial
from pre_process import PreProcess as pre

# stylesheet
# -------------------------------------------------------------------------------------------
Buttonstyle = "display: inline-block;" \
              "padding: 8px 16px;" \
              "font-size: 12px;" \
              "cursor: pointer;" \
              "text-align: center;" \
              "text-decoration: none;" \
              "outline: none;" \
              "color: black;" \
              "background-color: #FFFFFF;" \
              "border: none;" \
              "border-radius: 7px;" \
              "box-shadow: 0px 9px #999;"

groupboxStyle = "background-image: url(icon/kavica.png);" \
                "background-attachment: fixed;" \
                "background-repeat: no-repeat;" \
                "background-position: center;" \
                "border: 0px solid;" \
                "border-radius: 15px;"


# -------------------------------------------------------------------------------------------


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'KAVICA GUI'
        self.left = 10
        self.top = 10
        self.width = 320
        self.height = 240
        self.setFixedSize(400, 270)
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # Main menu
        mainMenu = self.menuBar()
        kavicaMenu = mainMenu.addMenu('Kavica')
        toolsMenu = mainMenu.addMenu('Tools')
        helpMenu = mainMenu.addMenu('Help')

        self.frames_widget = Frames(parent=self)
        self.setCentralWidget(self.frames_widget)

        self.show()


class Frames(QWidget):
    def __init__(self, parent=None):
        super(Frames, self).__init__(parent)
        grid = QGridLayout()
        grid.setColumnStretch(0, 1)
        grid.addWidget(self.create_left_frame(), 0, 0)
        grid.addWidget(self.create_right_frame(), 0, 1)
        self.setLayout(grid)


    def create_left_frame(self):
        groupBox = QGroupBox()
        groupBox.setStyleSheet(groupboxStyle)

        vbox = QVBoxLayout()
        vbox.addStretch(1)
        groupBox.setLayout(vbox)
        return groupBox

    def create_right_frame(self):
        groupBox = QGroupBox("Applications")

        buttonExplorer = QPushButton('Explorer', self)
        buttonExplorer.setStyleSheet(Buttonstyle)
        buttonExplorer.clicked.connect(partial(self.on_click, 1))

        buttonPreProcessing = QPushButton('Pre-Process', self)
        buttonPreProcessing.setStyleSheet(Buttonstyle)
        buttonPreProcessing.clicked.connect(partial(self.on_click, 2))

        buttonClustering = QPushButton('Cluster', self)
        buttonClustering.setStyleSheet(Buttonstyle)
        buttonClustering.clicked.connect(partial(self.on_click, 3))

        buttonPostProcessing = QPushButton('Post-Process', self)
        buttonPostProcessing.setStyleSheet(Buttonstyle)
        buttonPostProcessing.clicked.connect(partial(self.on_click, 4))

        buttonReserved = QPushButton('Reserved', self)
        buttonReserved.setStyleSheet(Buttonstyle)
        buttonReserved.clicked.connect(partial(self.on_click, 5))

        vbox = QVBoxLayout()
        vbox.addWidget(buttonExplorer)
        vbox.addWidget(buttonPreProcessing)
        vbox.addWidget(buttonClustering)
        vbox.addWidget(buttonPostProcessing)
        vbox.addWidget(buttonReserved)
        vbox.addStretch(1)
        groupBox.setLayout(vbox)

        return groupBox

    @pyqtSlot()
    def on_click(self, comand=None):
        print(comand)
        self.nd = pre()
        self.nd.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    clock = App()
    clock.show()
    sys.exit(app.exec_())
