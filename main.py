import sys
from Core.Rnn1Core import *
from Core.Rnn2Core import *
from GUI.VisualizationModule import *
from GUI.MainWindow import Ui_MainWindow
from PyQt5.QtCore import pyqtSignal
from Core.Params import *
import matplotlib.pyplot as plt

class GUI(QtWidgets.QMainWindow):

    signalRnn1ParamsChanged = pyqtSignal(str)
    signalRnn1ProcessingParamsChanged = pyqtSignal(str)
    signalRnn2ParamsChanged = pyqtSignal(str)
    signalRnn2ProcessingParamsChanged = pyqtSignal(str)
    signalDataToRnn2 = pyqtSignal(dict)

    def __init__(self):
        super(GUI, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        file_path = 'settings_common.ini'
        params = CommonParams(file_path)
        self.params = params

        self.restoreState(params.windowState)
        self.restoreGeometry(params.geometry)
        self.ui.tabWidget_rnn1.setCurrentIndex(self.params.tab_rnn1_index)
        self.ui.tabWidget_rnn2.setCurrentIndex(self.params.tab_rnn2_index)

        # rnn1
        self.thread1 = QtCore.QThread()
        rnn1_params = Rnn1Params('settings_rnn1.ini', params)
        self.rnn1 = Rnn1Core(params, rnn1_params)

        # rnn2
        self.thread2 = QtCore.QThread()
        rnn2_params = Rnn2Params('settings_rnn2.ini', params)
        self.rnn2 = Rnn2Core(params, rnn2_params)

        self.rnn2.signalFinished.connect(self.rnn1.rnn2Finished, QtCore.Qt.QueuedConnection)

        if self.params.draw_layers:
            # init visualization module
            self.graph_scene_rnn1_lr1 = GraphScene(0, params.L, params.M, params.d, params.q, self.rnn1.route)
            self.ui.graphicsView_rnn1_lr1.setScene(self.graph_scene_rnn1_lr1)
            self.ui.graphicsView_rnn1_lr1.horizontalScrollBar().rangeChanged.connect(
                lambda i1, i2: self.graphicsViewScrollChanged(i1, i2), QtCore.Qt.QueuedConnection)
            self.ui.graphicsView_rnn1_lr1.verticalScrollBar().rangeChanged.connect(
                lambda i1, i2: self.graphicsViewScrollChanged(i1, i2), QtCore.Qt.QueuedConnection)

            self.graph_scene_rnn1_lr2 = GraphScene(1, params.L, params.M, params.d, params.q, self.rnn1.route)
            self.ui.graphicsView_rnn1_lr2.setScene(self.graph_scene_rnn1_lr2)
            self.ui.graphicsView_rnn1_lr2.horizontalScrollBar().rangeChanged.connect(
                lambda i1, i2: self.graphicsViewScrollChanged(i1, i2), QtCore.Qt.QueuedConnection)
            self.ui.graphicsView_rnn1_lr2.verticalScrollBar().rangeChanged.connect(
                lambda i1, i2: self.graphicsViewScrollChanged(i1, i2), QtCore.Qt.QueuedConnection)

            self.graph_scene_rnn2_lr1 = GraphScene(0, params.L, params.M, params.d, params.q, self.rnn2.route)
            self.ui.graphicsView_rnn2_lr1.setScene(self.graph_scene_rnn2_lr1)
            self.ui.graphicsView_rnn2_lr1.horizontalScrollBar().rangeChanged.connect(
                lambda i1, i2: self.graphicsViewScrollChanged(i1, i2), QtCore.Qt.QueuedConnection)
            self.ui.graphicsView_rnn2_lr1.verticalScrollBar().rangeChanged.connect(
                lambda i1, i2: self.graphicsViewScrollChanged(i1, i2), QtCore.Qt.QueuedConnection)

            self.graph_scene_rnn2_lr2 = GraphScene(1, params.L, params.M, params.d, params.q, self.rnn2.route)
            self.ui.graphicsView_rnn2_lr2.setScene(self.graph_scene_rnn2_lr2)
            self.ui.graphicsView_rnn2_lr2.horizontalScrollBar().rangeChanged.connect(
                lambda i1, i2: self.graphicsViewScrollChanged(i1, i2), QtCore.Qt.QueuedConnection)
            self.ui.graphicsView_rnn2_lr2.verticalScrollBar().rangeChanged.connect(
                lambda i1, i2: self.graphicsViewScrollChanged(i1, i2), QtCore.Qt.QueuedConnection)

            self.rnn1.signalVisualize.connect(lambda a1: self.graph_scene_rnn1_lr1.drawGraphic(a1), QtCore.Qt.QueuedConnection)
            self.rnn1.signalVisualize.connect(lambda a1: self.graph_scene_rnn1_lr2.drawGraphic(a1), QtCore.Qt.QueuedConnection)
            self.graph_scene_rnn1_lr1.signalDrawed.connect(self.rnn1.drawed, QtCore.Qt.QueuedConnection)
            self.graph_scene_rnn1_lr2.signalDrawed.connect(self.rnn1.drawed, QtCore.Qt.QueuedConnection)
            self.rnn1.signalClearVisualize.connect(self.graph_scene_rnn1_lr1.clearGraphic, QtCore.Qt.QueuedConnection)
            self.rnn1.signalClearVisualize.connect(self.graph_scene_rnn1_lr2.clearGraphic, QtCore.Qt.QueuedConnection)

            self.rnn2.signalVisualize.connect(lambda a1: self.graph_scene_rnn2_lr1.drawGraphic(a1), QtCore.Qt.QueuedConnection)
            self.rnn2.signalVisualize.connect(lambda a1: self.graph_scene_rnn2_lr2.drawGraphic(a1), QtCore.Qt.QueuedConnection)
            self.graph_scene_rnn2_lr1.signalDrawed.connect(self.rnn2.drawed, QtCore.Qt.QueuedConnection)
            self.graph_scene_rnn2_lr2.signalDrawed.connect(self.rnn2.drawed, QtCore.Qt.QueuedConnection)
            self.rnn2.signalClearVisualize.connect(self.graph_scene_rnn2_lr1.clearGraphic, QtCore.Qt.QueuedConnection)
            self.rnn2.signalClearVisualize.connect(self.graph_scene_rnn2_lr2.clearGraphic, QtCore.Qt.QueuedConnection)

        # connecting
        self.ui.comboBox_ProcessingType.setCurrentText(self.params.processing_type)
        self.ui.predictStepsNum.setValue(self.params.predictStepsNum)
        self.ui.novFiltWeightsGain.setValue(self.params.novFiltWeightsGain)
        self.ui.novFiltDetectBorder.setValue(self.params.novFiltDetectBorder)
        self.ui.novFiltStepsNum.setValue(self.params.novFiltStepsNum)
        self.ui.comboBox_ProcessingType.currentTextChanged.connect(self.processingParamsChanged)
        self.ui.predictStepsNum.valueChanged.connect(self.processingParamsChanged)
        self.ui.novFiltWeightsGain.valueChanged.connect(self.processingParamsChanged)
        self.ui.novFiltDetectBorder.valueChanged.connect(self.processingParamsChanged)
        self.ui.novFiltStepsNum.valueChanged.connect(self.processingParamsChanged)
        self.processingParamsChanged()

        self.ui.pushButton_CopyModel.clicked.connect(self.rnn1.copyModel, QtCore.Qt.QueuedConnection)
        self.rnn1.signalModel.connect(lambda a1: self.Rnn1ToRnn2(a1), QtCore.Qt.QueuedConnection)
        self.signalDataToRnn2.connect(lambda a1: self.rnn2.pasteModel(a1), QtCore.Qt.QueuedConnection)

        if params.continuous_mode:
            self.ui.pushButton_rnn1_next.setEnabled(False)
            self.ui.pushButton_rnn2_next.setEnabled(False)

        self.ui.rnn1_alpha.setValue(self.rnn1.rnn_params.alpha)
        self.ui.rnn1_h.setValue(self.rnn1.rnn_params.h)
        self.ui.rnn1_gInc.setValue(self.rnn1.rnn_params.gInc)
        self.ui.rnn1_gDec.setValue(self.rnn1.rnn_params.gDec)
        self.ui.rnn1_flag_learn.setChecked(self.rnn1.rnn_params.flag_learning)
        self.ui.rnn1_flag_clear_learning.setChecked(self.rnn1.rnn_params.flag_clear_learning)
        self.ui.rnn1_input_data_file_path.setText(self.rnn1.rnn_params.input_data_filename)
        self.ui.rnn1_borderType.setCurrentText(self.rnn1.rnn_params.border_type)
        self.ui.rnn1_borderConstValue.setValue(self.rnn1.rnn_params.border_Const_value)
        self.ui.rnn1_borderConcurrentWinners.setValue(self.rnn1.rnn_params.border_Concurrent_winners)

        self.ui.pushButton_rnn1_refreshParams.clicked.connect(self.refreshRnnParams, QtCore.Qt.QueuedConnection)
        self.signalRnn1ParamsChanged.connect(lambda a1: self.rnn1.refreshParams(a1), QtCore.Qt.QueuedConnection)

        self.ui.pushButton_rnn1_begin.clicked.connect(self.rnn1.startProcessSignals, QtCore.Qt.QueuedConnection)
        self.ui.pushButton_rnn1_next.clicked.connect(self.rnn1.processSignals, QtCore.Qt.QueuedConnection)
        self.ui.pushButton_rnn1_end.clicked.connect(self.rnn1.finishProcessSignals, QtCore.Qt.QueuedConnection)
        self.ui.pushButton_rnn1_clear.clicked.connect(self.rnn1.clearRnn, QtCore.Qt.QueuedConnection)

        self.ui.pushButton_rnn2_next.clicked.connect(self.rnn2.processSignals, QtCore.Qt.QueuedConnection)
        self.ui.pushButton_rnn2_end.clicked.connect(self.rnn2.finishProcessSignals, QtCore.Qt.QueuedConnection)

        self.ui.rnn2_alpha.setValue(self.rnn2.rnn_params.alpha)
        self.ui.rnn2_h.setValue(self.rnn2.rnn_params.h)
        self.ui.rnn2_borderType.setCurrentText(self.rnn2.rnn_params.border_type)
        self.ui.rnn2_borderConstValue.setValue(self.rnn2.rnn_params.border_Const_value)
        self.ui.rnn2_borderConcurrentWinners.setValue(self.rnn2.rnn_params.border_Concurrent_winners)

        self.ui.pushButton_rnn2_refreshParams.clicked.connect(self.refreshRnnParams, QtCore.Qt.QueuedConnection)
        self.signalRnn2ParamsChanged.connect(lambda a1: self.rnn2.refreshParams(a1), QtCore.Qt.QueuedConnection)

        self.signalRnn1ProcessingParamsChanged.connect(lambda a1: self.rnn1.refreshProcessingParams(a1), QtCore.Qt.QueuedConnection)
        self.signalRnn2ProcessingParamsChanged.connect(lambda a1: self.rnn2.refreshProcessingParams(a1), QtCore.Qt.QueuedConnection)

        self.rnn1.signalPlot.connect(lambda a1: self.plot(a1))

        # threads
        self.rnn1.moveToThread(self.thread1)
        self.thread1.start()
        self.rnn2.moveToThread(self.thread2)
        self.thread2.start()

    def Rnn1ToRnn2(self, rnn1_to_rnn2_data):
        #print('rnn1_to_rnn2_data')

        if self.ui.comboBox_ProcessingType.currentText() == 'Predict':
            rnn1_to_rnn2_data['processing_type'] = 'Predict'
            rnn1_to_rnn2_data['predictStepsNum'] = self.ui.predictStepsNum.value()
        elif self.ui.comboBox_ProcessingType.currentText() == 'Novelty filter':
            rnn1_to_rnn2_data['processing_type'] = 'Novelty filter'
            rnn1_to_rnn2_data['novFiltStepsNum'] = self.ui.novFiltStepsNum.value()
        else:
            print('selected not provide processing type')
            return

        self.signalDataToRnn2.emit(rnn1_to_rnn2_data)

    def plot(self, data):
        plt.plot(data)
        plt.show()

    def graphicsViewScrollChanged(self, _min, _max):
        sender = self.sender()
        if sender == self.ui.graphicsView_rnn1_lr1.horizontalScrollBar():
            [pixelSize_height, pixelSize_wigth, coord_y, coord_x] = self.graph_scene_rnn1_lr1.getScrollParams()
            delta = coord_x - pixelSize_wigth / 2
            value = delta + _max / 2
            if value < _min:
                value = _min
            if value > _max:
                value = _max
            self.ui.graphicsView_rnn1_lr1.horizontalScrollBar().setValue(int(value))
        elif sender == self.ui.graphicsView_rnn1_lr1.verticalScrollBar():
            [pixelSize_height, pixelSize_wigth, coord_y, coord_x] = self.graph_scene_rnn1_lr1.getScrollParams()
            delta = coord_y - pixelSize_height / 2
            value = delta + _max / 2
            if value < _min:
                value = _min
            if value > _max:
                value = _max
            self.ui.graphicsView_rnn1_lr1.horizontalScrollBar().setValue(int(value))
        elif sender == self.ui.graphicsView_rnn1_lr2.horizontalScrollBar():
            [pixelSize_height, pixelSize_wigth, coord_y, coord_x] = self.graph_scene_rnn1_lr2.getScrollParams()
            delta = coord_x - pixelSize_wigth / 2
            value = delta + _max / 2
            if value < _min:
                value = _min
            if value > _max:
                value = _max
            self.ui.graphicsView_rnn1_lr2.horizontalScrollBar().setValue(int(value))
        elif sender == self.ui.graphicsView_rnn1_lr2.verticalScrollBar():
            [pixelSize_height, pixelSize_wigth, coord_y, coord_x] = self.graph_scene_rnn1_lr2.getScrollParams()
            delta = coord_y - pixelSize_height / 2
            value = delta + _max / 2
            if value < _min:
                value = _min
            if value > _max:
                value = _max
            self.ui.graphicsView_rnn1_lr2.horizontalScrollBar().setValue(int(value))
        elif sender == self.ui.graphicsView_rnn2_lr1.horizontalScrollBar():
            [pixelSize_height, pixelSize_wigth, coord_y, coord_x] = self.graph_scene_rnn2_lr1.getScrollParams()
            delta = coord_x - pixelSize_wigth / 2
            value = delta + _max / 2
            if value < _min:
                value = _min
            if value > _max:
                value = _max
            self.ui.graphicsView_rnn2_lr1.horizontalScrollBar().setValue(int(value))
        elif sender == self.ui.graphicsView_rnn2_lr1.verticalScrollBar():
            [pixelSize_height, pixelSize_wigth, coord_y, coord_x] = self.graph_scene_rnn2_lr1.getScrollParams()
            delta = coord_y - pixelSize_height / 2
            value = delta + _max / 2
            if value < _min:
                value = _min
            if value > _max:
                value = _max
            self.ui.graphicsView_rnn2_lr1.horizontalScrollBar().setValue(int(value))
        elif sender == self.ui.graphicsView_rnn2_lr1.horizontalScrollBar():
            [pixelSize_height, pixelSize_wigth, coord_y, coord_x] = self.graph_scene_rnn2_lr2.getScrollParams()
            delta = coord_x - pixelSize_wigth / 2
            value = delta + _max / 2
            if value < _min:
                value = _min
            if value > _max:
                value = _max
            self.ui.graphicsView_rnn2_lr1.horizontalScrollBar().setValue(int(value))
        elif sender == self.ui.graphicsView_rnn2_lr1.verticalScrollBar():
            [pixelSize_height, pixelSize_wigth, coord_y, coord_x] = self.graph_scene_rnn2_lr2.getScrollParams()
            delta = coord_y - pixelSize_height / 2
            value = delta + _max / 2
            if value < _min:
                value = _min
            if value > _max:
                value = _max
            self.ui.graphicsView_rnn2_lr1.horizontalScrollBar().setValue(int(value))


    def refreshRnnParams(self):
        sender = self.sender()
        if sender == self.ui.pushButton_rnn1_refreshParams:
            snd = {'alpha': self.ui.rnn1_alpha.value(),
                   'h': self.ui.rnn1_h.value(),
                   'gInc': self.ui.rnn1_gInc.value(),
                   'gDec': self.ui.rnn1_gDec.value(),
                   'flag_learning': self.ui.rnn1_flag_learn.isChecked(),
                   'flag_clear_learning': self.ui.rnn1_flag_clear_learning.isChecked(),
                   'input_data_filename': self.ui.rnn1_input_data_file_path.text(),
                   'border_type': self.ui.rnn1_borderType.currentText(),
                   'border_Const_value': self.ui.rnn1_borderConstValue.value(),
                   'border_Concurrent_winners': self.ui.rnn1_borderConcurrentWinners.value(),
                    }
            self.signalRnn1ParamsChanged.emit(json.dumps(snd))
        elif sender == self.ui.pushButton_rnn2_refreshParams:
            snd = {'alpha': self.ui.rnn2_alpha.value(),
                   'h': self.ui.rnn2_h.value(),
                   'border_type': self.ui.rnn2_borderType.currentText(),
                   'border_Const_value': self.ui.rnn2_borderConstValue.value(),
                   'border_Concurrent_winners': self.ui.rnn2_borderConcurrentWinners.value(),
                   }
            self.signalRnn2ParamsChanged.emit(json.dumps(snd))

    def processingParamsChanged(self):
        snd = {'processing_type': self.ui.comboBox_ProcessingType.currentText(),
               'predictStepsNum': self.ui.predictStepsNum.value(),
               'novFiltWeightsGain': self.ui.novFiltWeightsGain.value(),
               'novFiltDetectBorder': self.ui.novFiltDetectBorder.value(),
               'novFiltStepsNum': self.ui.novFiltStepsNum.value()
               }
        self.signalRnn1ProcessingParamsChanged.emit(json.dumps(snd))
        self.signalRnn2ProcessingParamsChanged.emit(json.dumps(snd))

        if self.ui.comboBox_ProcessingType.currentText() == 'Novelty filter':
            self.ui.pushButton_CopyModel.setEnabled(False)
        else:
            self.ui.pushButton_CopyModel.setEnabled(True)

        self.params.processing_type = self.ui.comboBox_ProcessingType.currentText()
        self.params.predictStepsNum = self.ui.predictStepsNum.value()
        self.params.novFiltWeightsGain = self.ui.novFiltWeightsGain.value()
        self.params.novFiltDetectBorder = self.ui.novFiltDetectBorder.value()
        self.params.novFiltStepsNum = self.ui.novFiltStepsNum.value()

    def closeEvent(self, event):
        self.ui.tabWidget_rnn1.currentIndex()

        self.params.windowState = self.saveState()
        self.params.geometry = self.saveGeometry()

        self.params.tab_rnn1_index = self.ui.tabWidget_rnn1.currentIndex()
        self.params.tab_rnn2_index = self.ui.tabWidget_rnn2.currentIndex()

        self.params.rewrite()

        super().closeEvent(event)

if __name__ == '__main__':

    app = QtWidgets.QApplication([])
    application = GUI()
    application.show()
    sys.exit(app.exec())
