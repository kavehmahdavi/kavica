#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KAVICA GUI
"""
# Author: Kaveh Mahdavi <kavehmahdavi74@gmail.com>
# License: BSD 3 clause
# last update: 14/12/2018


import sys
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import missingno as msno
from pyqtgraph import QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QMainWindow,
                             QHBoxLayout,
                             QTableWidget,
                             QGroupBox,
                             QPushButton,
                             QTableWidgetItem,
                             QFormLayout,
                             QLabel,
                             QTabWidget,
                             QStyle,
                             QVBoxLayout,
                             QAbstractScrollArea,
                             QWidget,
                             QGridLayout,
                             QApplication)

# -------------------------------------------------------------------------------------------
# stylesheet
# -------------------------------------------------------------------------------------------
'''
app.setStyleSheet("""
        QTabWidget::pane { /* The tab widget frame */
            border-top: 2px solid #E6E6FA;
        }

        QTabWidget::tab-bar {
            left: 5px; /* move to the right by 5px */
        }

        /* Style the tab using the tab sub-control. Note that
            it reads QTabBar _not_ QTabWidget */
        QTabBar::tab {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                        stop: 0 #E1E1E1, stop: 0.4 #DDDDDD,
                                        stop: 0.5 #D8D8D8, stop: 1.0 #D3D3D3);
            border: 2px solid #C4C4C3;
            border-bottom-color: #C2C7CB; /* same as the pane color */
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            min-width: 8ex;
            padding: 2px;
        }

        QTabBar::tab:selected, QTabBar::tab:hover {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                        stop: 0 #fafafa, stop: 0.4 #f4f4f4,
                                        stop: 0.5 #e7e7e7, stop: 1.0 #fafafa);
            background-color: #E6E6FA;
        }

        QTabBar::tab:selected {
            border-color: #9B9B9B;
            border-bottom-color: #E6E6FA; /* same as pane color */
        }

        QTabBar::tab:!selected {
            margin-top: 2px; /* make non-selected tabs look smaller */
        }
""")
'''


# -------------------------------------------------------------------------------------------
def damp(data, pers=0.2):
    x = [random.randint(1, data.shape[0] - 1) for p in range(0, round(data.shape[0] * pers))]
    y = [random.randint(1, data.shape[1] - 1) for p in range(0, round(data.shape[0] * pers))]
    for i, j in zip(x, y):
        data.iloc[i, j] = np.nan
    return data


class PreProcess(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'Kavica Pre-Process'
        self.left = 100
        self.top = 100
        self.width = 940
        self.height = 670
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.table_widget = MyTableWidget(self)
        self.setCentralWidget(self.table_widget)
        self.show()


class MyTableWidget(QWidget):

    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.layout = QVBoxLayout(self)

        # Initialize tab screen
        # --------------------------------------------------------
        self.tabs = QTabWidget()
        self.Imputation = QWidget()
        self.Outliers = QWidget()
        self.Transform = QWidget()
        self.FeatureSelection = QWidget()
        self.tabs.resize(300, 200)

        # Add tabs
        # --------------------------------------------------------
        self.tabs.addTab(self.Imputation, "Imputation")
        self.tabs.addTab(self.Outliers, "Outliers")
        self.tabs.addTab(self.Transform, "Transform")
        self.tabs.addTab(self.FeatureSelection, "Feature Selection")

        # Create tabs
        # --------------------------------------------------------
        self.create_imputation_tab()
        self.create_outliers_tab()

        # Add tabs to widget
        # --------------------------------------------------------
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

    def create_imputation_tab(self):
        self.Imputation.layout = QVBoxLayout(self)
        self.Imputation.layout.addWidget(ImputationTabObject())
        self.Imputation.setLayout(self.Imputation.layout)

    def create_outliers_tab(self):
        # It is needed to move to the class outlier.
        self.Outliers.layout = QVBoxLayout(self)

        button_open_file = QPushButton('Open file...', self)
        button_open_config = QPushButton('Open config...', self)
        button_open_file.setMaximumSize(120, 100)
        button_open_config.setMaximumSize(120, 100)
        self.Outliers.layout.addWidget(button_open_file)
        self.Outliers.layout.addWidget(button_open_config)

        self.Outliers.layout.addWidget(OutliersTabObject())
        self.Outliers.setLayout(self.Outliers.layout)


class ImputationTabObject(QWidget):
    def __init__(self, parent=None):
        super(ImputationTabObject, self).__init__(parent)
        self.feature_table_widget = None
        grid = QGridLayout()
        grid.addWidget(self.file_group(), 0, 0, 1, 2)
        grid.addWidget(self.filter_group(), 1, 0, 1, 2)
        grid.addWidget(self.features_group(), 2, 0)
        grid.addWidget(self.missing_group(), 2, 1)
        self.setLayout(grid)

    def file_group(self):
        # Group box: file.
        # --------------------------------------------------------------------------------------------------------------
        formGroupBox = QGroupBox("File")

        # Form: file.
        # --------------------------------------------------------------------------------------------------------------
        layout = QFormLayout()

        # Horizontal box: file loading.
        # --------------------------------------------------------------------------------------------------------------
        hbox_file = QHBoxLayout()

        button_choose_file = QPushButton('Trace ', self)
        button_choose_file.setIcon(self.style().standardIcon(getattr(QStyle,
                                                                     'SP_FileDialogNewFolder')))
        labelFile = QLabel('Lulesh_27p.prv', self)
        labelFile.setStyleSheet("background-color: white; border: 1px inset grey;")

        hbox_file.addWidget(button_choose_file)
        hbox_file.addWidget(labelFile, 1)
        hbox_file.addStretch()

        # Horizontal box: configuration loading.
        # --------------------------------------------------------------------------------------------------------------
        hbox_config = QHBoxLayout()

        button_choose_config = QPushButton('Config', self)
        button_choose_config.setIcon(self.style().standardIcon(getattr(QStyle,
                                                                       'SP_FileDialogNewFolder')))
        label_config = QLabel('Config_27p.json', self)
        label_config.setStyleSheet("background-color: white; border: 1px inset grey;")

        button_load = QPushButton('Load')
        button_load.setIcon(self.style().standardIcon(getattr(QStyle,
                                                              'SP_FileDialogStart')))

        hbox_config.addWidget(button_choose_config)
        hbox_config.addWidget(label_config, 1)
        hbox_file.addWidget(button_load)
        hbox_config.addStretch()
        # --------------------------------------------------------------------------------------------------------------

        layout.addRow(hbox_config)
        layout.addRow(hbox_file)

        formGroupBox.setLayout(layout)
        return formGroupBox

    def filter_group(self):
        # Group box: file.
        # --------------------------------------------------------------------------------------------------------------
        filterBox = QGroupBox("Filter")

        # Horizontal box: filter selection.
        # --------------------------------------------------------------------------------------------------------------
        hbox_filter = QHBoxLayout()

        button_choose = QPushButton('', self)
        button_choose.setIcon(self.style().standardIcon(getattr(QStyle,
                                                                'SP_FileDialogNewFolder')))
        labelFilter = QLabel('Duration_filter.json', self)
        labelFilter.setStyleSheet("background-color: white; border: 1px inset grey;")

        button_apply = QPushButton('Apply', self)
        button_apply.setIcon(self.style().standardIcon(getattr(QStyle,
                                                               'SP_DialogApplyButton')))
        button_stop = QPushButton('Stop', self)
        button_stop.setIcon(self.style().standardIcon(getattr(QStyle,
                                                              'SP_DialogCancelButton')))

        hbox_filter.addWidget(button_choose)
        hbox_filter.addWidget(labelFilter, 1)
        hbox_filter.addStretch()
        hbox_filter.addWidget(button_apply)
        hbox_filter.addWidget(button_stop)

        # --------------------------------------------------------------------------------------------------------------
        filterBox.setLayout(hbox_filter)
        return filterBox

    def features_group(self):
        # Group box: feature table.
        # --------------------------------------------------------------------------------------------------------------
        groupBox = QGroupBox("Features")

        # Vertical box: feature table.
        # --------------------------------------------------------------------------------------------------------------
        vbox_feature_table = QVBoxLayout()

        # Table: create a table of feature
        # TODO: it is needed to read from the config.
        featureList = ["IPC", "PAPI_TOT_INS", "PAPI_TOT_CYC",
                       "PAPI_L1_DCM", "PAPI_L2_DCM", "PAPI_BR_INS",
                       "PAPI_BR_MSP", "RESOURCE_STALLS_SB", "RESOURCE_STALLS_ROB",
                       "Caller_at_level_1", "Caller_at_level_2", "Caller_at_level_3",
                       "Caller_line_at_level_1", "Caller_line_at_level_2", "Caller_line_at_level_3"]
        self.feature_table_widget = QTableWidget()
        self.feature_table_widget.setAlternatingRowColors(True)
        self.feature_table_widget.setRowCount(len(featureList))
        self.feature_table_widget.setColumnCount(2)
        self.feature_table_widget.setHorizontalHeaderLabels(["Name", "Feature"])

        for index, featureItem in enumerate(featureList):
            chkBoxItem = QTableWidgetItem()
            chkBoxItem.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            chkBoxItem.setCheckState(QtCore.Qt.Unchecked)
            self.feature_table_widget.setItem(index, 0, chkBoxItem)
            self.feature_table_widget.setItem(index, 1, QTableWidgetItem(featureItem))

        self.feature_table_widget.move(0, 0)
        self.feature_table_widget.resizeColumnsToContents()
        self.set_table_width()

        # table size policy
        self.feature_table_widget.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)

        # table selection change
        self.feature_table_widget.doubleClicked.connect(self.on_click)

        vbox_feature_table.addWidget(self.feature_table_widget)
        vbox_feature_table.addStretch(1)

        # --------------------------------------------------------------------------------------------------------------
        groupBox.setLayout(vbox_feature_table)
        return groupBox

    def set_table_width(self):
        table_width = self.feature_table_widget.verticalHeader().width()
        table_width += self.feature_table_widget.horizontalHeader().length()
        if self.feature_table_widget.verticalScrollBar().isVisible():
            table_width += self.feature_table_widget.verticalScrollBar().width()
        table_width += self.feature_table_widget.frameWidth() * 2
        self.feature_table_widget.setFixedWidth(table_width)

    @staticmethod
    def missing_group():

        def _missing_plot():
            df = pd.read_csv('../../kavica/parser/source.csv')
            df = damp(df)
            missing_plot = msno.matrix(df, figsize=(8, 5), color=(0.580, 0.180, 0.196))
            missing_plot.xaxis.set_tick_params(labelsize=6)
            missing_plot.yaxis.set_tick_params(labelsize=10)
            plt.savefig('missing_plot.png')

        # Group box: Missing values.
        # --------------------------------------------------------------------------------------------------------------
        groupBox = QGroupBox("Missing Values")

        # Vertical box: feature table.
        # --------------------------------------------------------------------------------------------------------------
        vbox_missing_plot = QVBoxLayout()

        # Generate the missing plot.
        _missing_plot()

        label_missing_plot = QLabel()
        label_missing_plot.setAlignment(QtCore.Qt.AlignCenter)
        label_missing_plot.setPixmap(QPixmap('missing_plot.png'))

        # TODO: run plot by click, Just some button connected to `plot` method
        button_plot = QPushButton('Plot')

        vbox_missing_plot.addWidget(label_missing_plot)
        vbox_missing_plot.addWidget(button_plot)
        vbox_missing_plot.addStretch(1)

        # --------------------------------------------------------------------------------------------------------------
        groupBox.setLayout(vbox_missing_plot)
        return groupBox

    @pyqtSlot()
    def on_click(self):
        print("\n")
        for currentQTableWidgetItem in self.feature_table_widget.selectedItems():
            print(currentQTableWidgetItem.row(),
                  currentQTableWidgetItem.column(),
                  currentQTableWidgetItem.text())


class OutliersTabObject(QWidget):
    def __init__(self, parent=None):
        super(OutliersTabObject, self).__init__(parent)
        grid = QGridLayout()
        grid.addWidget(self.filter_group(), 0, 0, 1, 2)
        grid.addWidget(self.features_group(), 1, 0)
        grid.addWidget(self.missing_group(), 1, 1)
        self.setLayout(grid)

    def filter_group(self):
        filterBox = QGroupBox("Filter")
        button_choose = QPushButton('Choose', self)
        labelFilter = QLabel('None', self)
        labelFilter.setStyleSheet("background-color: white; border: 1px inset grey;")
        button_apply = QPushButton('Apply', self)
        button_stop = QPushButton('Stop', self)
        hbox = QHBoxLayout()
        hbox.addWidget(button_choose)
        hbox.addWidget(labelFilter, 1)
        hbox.addStretch()
        hbox.addWidget(button_apply)
        hbox.addWidget(button_stop)
        filterBox.setLayout(hbox)
        return filterBox

    def features_group(self):
        groupBox = QGroupBox("Features")

        # TODO: it is needed to read from the config.
        # Create table
        featureList = ["IPC", "PAPI_TOT_INS", "PAPI_TOT_CYC", "PAPI_L1_DCM", "PAPI_L2_DCM",
                       "PAPI_BR_INS", "PAPI_BR_MSP", "RESOURCE_STALLS_SB", "RESOURCE_STALLS_ROB",
                       "Caller_at_level_1", "Caller_at_level_2", "Caller_at_level_3",
                       "Caller_line_at_level_1", "Caller_line_at_level_2", "Caller_line_at_level_3"]
        self.feature_table_widget = QTableWidget()
        self.feature_table_widget.setAlternatingRowColors(True)
        self.feature_table_widget.setRowCount(len(featureList))
        self.feature_table_widget.setColumnCount(2)
        self.feature_table_widget.setHorizontalHeaderLabels(["Name", "Feature"])

        for index, featureItem in enumerate(featureList):
            chkBoxItem = QTableWidgetItem()
            chkBoxItem.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            chkBoxItem.setCheckState(QtCore.Qt.Unchecked)
            self.feature_table_widget.setItem(index, 0, chkBoxItem)
            self.feature_table_widget.setItem(index, 1, QTableWidgetItem(featureItem))

        self.feature_table_widget.move(0, 0)
        self.feature_table_widget.resizeColumnsToContents()
        self.set_table_width()

        # size policy
        self.feature_table_widget.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)

        # table selection change
        self.feature_table_widget.doubleClicked.connect(self.on_click)

        vbox = QVBoxLayout()
        vbox.addWidget(self.feature_table_widget)
        vbox.addStretch(1)
        groupBox.setLayout(vbox)
        return groupBox

    def set_table_width(self):
        width = self.feature_table_widget.verticalHeader().width()
        width += self.feature_table_widget.horizontalHeader().length()
        if self.feature_table_widget.verticalScrollBar().isVisible():
            width += self.feature_table_widget.verticalScrollBar().width()
        width += self.feature_table_widget.frameWidth() * 2
        self.feature_table_widget.setFixedWidth(width)

    def missing_group(self):

        def read_data():
            self.df = pd.read_csv('../../kavica/parser/source1.csv')
            self.df = damp(self.df)
            self.df = self.df.loc[:, ["PAPI_TOT_INS", "PAPI_TOT_CYC", "PAPI_L1_DCM"]]

        def plot():
            ''' plot some random stuff '''
            read_data()
            data = self.df.as_matrix()
            my_label = ["PAPI_TOT_INS", "PAPI_TOT_CYC", "PAPI_L1_DCM"]

            # instead of ax.hold(False)
            self.figure.clear()

            # create an axis
            ax = self.figure.add_subplot(111)

            # notch shape box plot
            box_plot = ax.boxplot(data,
                                  vert=True,  # vertical box alignment
                                  patch_artist=True,  # fill with color
                                  labels=my_label)  # will be used to label x-ticks

            # set the colors.
            for box in box_plot['boxes']:
                # change outline color
                box.set(color='black', linewidth=2)
                # change fill color
                box.set(facecolor=(0.580, 0.180, 0.196))
                # change hatch
                box.set(hatch='/')

            ax.set_title('Outliers')

            # discards the old graph
            # ax.hold(False) # deprecated, see above

            # refresh canvas
            self.canvas.draw()

        groupBox = QGroupBox("Missing Values")

        # a figure instance to plot on
        self.figure = plt.figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # Just some button connected to `plot` method
        self.button = QPushButton('Plot')
        self.button.clicked.connect(plot)

        vbox = QVBoxLayout()
        vbox.addWidget(self.canvas)
        vbox.addWidget(self.button)
        vbox.addStretch(1)
        groupBox.setLayout(vbox)

        return groupBox

    @pyqtSlot()
    def on_click(self):
        print("\n")
        for currentQTableWidgetItem in self.feature_table_widget.selectedItems():
            print(currentQTableWidgetItem.row(), currentQTableWidgetItem.column(), currentQTableWidgetItem.text())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = PreProcess()
    sys.exit(app.exec_())
