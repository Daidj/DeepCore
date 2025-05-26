import sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic

from PyQt5.QtCore import pyqtSlot

import form
import add_dataset
import coreset
from controller import Controller

class MainWindow(QtWidgets.QMainWindow, form.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("CoreSet Selection Tool")  # 设置窗体主体
        self.initUI()  # 构造功能函数

    def initUI(self):
        self.actionAdd_DataSet.triggered.connect(self.show_add_dataset)
        self.actionCoreSet_Selection.triggered.connect(self.show_coreset_selection)
        self.show_coreset_selection()

    def show_add_dataset(self):
        add_dataset_form = AddDataSetWindow()

        self.setCentralWidget(add_dataset_form)

    def show_coreset_selection(self):
        coreset_selection_form = CoreSetWindow()
        self.setCentralWidget(coreset_selection_form)


class AddDataSetWindow(QtWidgets.QMainWindow, add_dataset.Ui_Dialog):
    def __init__(self):
        super(AddDataSetWindow, self).__init__()
        self.setupUi(self)
        self.initUI()

    def initUI(self):
        self.addButton.clicked.connect(self.add_dataset)

    def add_dataset(self):
        args = {}
        args['dataset'] = self.dataset.toPlainText()
        args['dataset type'] = self.dataSetType.currentText()
        args['number of classes'] = int(self.numberOfClasses.toPlainText())
        args['dataset path'] = self.dataSetPath.toPlainText()
        success = Controller.getController().add_dataset(args)
        if success:
            QtWidgets.QMessageBox.information(self, 'Success', 'Add The DataSet Successfully')
        else:
            QtWidgets.QMessageBox.critical(self, 'Error', 'The DataSet Exists')

class CoreSetWindow(QtWidgets.QMainWindow, coreset.Ui_Dialog):
    def __init__(self):
        super(CoreSetWindow, self).__init__()
        self.setupUi(self)
        self.initUI()

    def initUI(self):
        self.dataSetType.currentIndexChanged.connect(self.update_dataset_list)
        self.dataSetType.currentIndexChanged.connect(self.update_model_list)
        self.runButton.clicked.connect(self.on_run_buttion_clicked)
        self.stopButton.clicked.connect(self.on_stop_buttion_clicked)
        self.stopButton.setVisible(False)
        Controller.getController().logger.append(self.log)
        self.update_dataset_list()
        self.update_model_list()
        self.update_algorithm_list()
        self.Log.append('init finish')


    def update_dataset_list(self):
        current_value = self.dataSetType.currentText()
        self.dataSet.clear()
        self.dataSet.addItems(Controller.getController().get_dataset_list(current_value))

    def update_model_list(self):
        current_value = self.dataSetType.currentText()
        self.model.clear()
        self.model.addItems(Controller.getController().get_model_list(current_value))

    def update_algorithm_list(self):
        self.algorithm.clear()
        self.algorithm.addItems(Controller.getController().get_algorithm_list())

    def log(self, text):
        self.Log.append(text)

    @pyqtSlot()
    def on_run_buttion_clicked(self):  # 利用QT自带槽函数直接连接按钮
        self.log('Run:')
        self.log('============')
        args = {}
        args['--dataset'] = self.dataSet.currentText()
        args['--model'] = self.model.currentText()
        args['--selection'] = 'MOEA2' if self.algorithm.currentText() == 'MicroSearch' else self.algorithm.currentText()
        args['--fraction'] = float(self.fraction.toPlainText())
        args['--iter'] = int(self.iteration.toPlainText())
        args['--population'] = int(self.population.toPlainText())
        Controller.getController().set_output_directory(self.outputFolder.toPlainText())
        Controller.getController().run_algorithm(args)
        self.log('Algorithm Finished')
    def on_stop_buttion_clicked(self):
        pass

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec())

