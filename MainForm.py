# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import PyQt5
import PyQt5.Qsci  # 这是为了验证Qsci是否安装成功

import sys
from PyQt5 import QtCore, QtGui, QtWidgets

# Press the green button in the gutter to run the script.
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QIcon

import form

class MainWindow(QtWidgets.QMainWindow, form.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # self.setWindowIcon(QIcon("./IAIS.png"))
        self.setWindowTitle("CoreSet Selection Tool")  # 设置窗体主体
        self.initUI()  # 构造功能函数

    def initUI(self):
        self.dataSetType.currentIndexChanged.connect(self.update_dataset_list)
        self.dataSetType.currentIndexChanged.connect(self.update_model_list)
        self.dataSet.currentIndexChanged.connect(self.update_dataset_path_enable)

        # 此处添加功能连接函数
        # self.run_button.clicked.connect(self.bofang_video)  # 自定义按钮连接自定义槽函数
        # self.pushButton1.pressed.connect()   #不同的按钮点击方式
        # self.pushButton1.released.connect()

    def update_dataset_list(self):
        current_value = self.dataSetType.currentText()
        self.dataSet.clear()
        if current_value == 'Image':
            self.dataSet.addItems(['MNIST', 'CIFAR10', 'CIFAR100'])
        elif current_value == 'Text':
            self.dataSet.addItems(['SST-5'])
        elif current_value == 'Audio':
            self.dataSet.addItems(['UrbanSound8K'])
        self.dataSet.addItem('Custom')

    def update_model_list(self):
        current_value = self.dataSetType.currentText()
        self.model.clear()
        if current_value == 'Image':
            self.model.addItems(['LeNet-5', 'ResNet18'])
        elif current_value == 'Text':
            self.model.addItems(['TextCNN'])
        elif current_value == 'Audio':
            self.model.addItems(['TDNN'])

    def update_dataset_path_enable(self):
        current_value = self.dataSet.currentText()

        if current_value == 'Custom':

            self.dataSetPath.setEnabled(True)
        else:
            self.dataSetPath.setEnabled(False)

    @pyqtSlot()
    def on_run_buttion_clicked(self):  # 利用QT自带槽函数直接连接按钮
        self.Log.append("Run:")
        self.Log.append("<font color=\"#00FF00\">Start</font> ")  # 设置字体颜色
        self.Log.append("======================")

    def bofang_video(self):
        print("123")
        # self.textBrowser.append("点击按钮1！")
        # self.textBrowser.append("<font color=\"#FF0000\">点击按钮1！</font> ")  # 设置字体颜色
        # self.textBrowser.append("======================")



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = MainWindow()
    MainWindow.show()
    sys.exit(app.exec())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
