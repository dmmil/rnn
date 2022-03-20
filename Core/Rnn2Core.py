from Core.AbstractRnnCore import *


class Rnn2Core(AbstractRnnCore):

    signalFinished = pyqtSignal()

    def __init__(self, common_params: CommonParams, rnn_params):
        super(Rnn2Core, self).__init__(common_params, rnn_params)

        self.io_device = WordConnectionsIODevice(self.common_params.d, self.common_params.q)

    def emitNeuronesSinglePulses(self):
        for item in self.SSPs:
            if item == self.route[len(self.route)-1]:
                continue

            route_ssp_index = self.route.index(item)
            tmpIndexes = self.future_route_indexes[route_ssp_index]

            for id in range(self.common_params.d):
                for iq in range(self.common_params.q):
                    if self.neu_states[self.route.index(item), id, iq] == -1:
                        for ind in tmpIndexes:
                            if not self.route[ind] in self.future_SSPs:
                                continue
                            if ind == route_ssp_index + 1:
                                # only direct synapses
                                self.neu_current_values[route_ssp_index + 1,
                                                        np.logical_or(
                                                            self.neu_states[route_ssp_index, :, :] == -1,
                                                            self.neu_states[route_ssp_index, :, :] ==
                                                            self.common_params.refract_interval)] += 1
                            else:
                                self.neu_current_values[ind, :, :] += \
                                self.snp_k[route_ssp_index, id, iq, tmpIndexes.index(ind), :, :] * \
                                self.snp_b[route_ssp_index, tmpIndexes.index(ind)]

    def learnRnn(self):
        pass

    def analyzeRnnState(self):
        pass

    def pasteModel(self, model_dict):

        if len(self.SSPs):
            self.finishProcessSignals()

        self.flag_processing = True

        self.snp_k = model_dict['snp_k']
        self.SSPs = model_dict['SSPs']
        self.neu_states = model_dict['neu_states']
        self.sspTact = model_dict['sspTact']
        self.io_device.setIODeviceState(model_dict['io_device'])

        if model_dict['processing_type'] == 'Predict':
            self.io_device.modifyForPredict(model_dict['predictStepsNum'])
        elif model_dict['processing_type'] == 'Novelty filter':
            self.io_device.modifyForNoveltyFilter(self.common_params.rnn2DelayNumTacts, model_dict['novFiltStepsNum'])

        if self.common_params.draw_layers:
            if self.common_params.continuous_mode:
                self.flag_draw_answer_is_needed = 2
            self.signalVisualize.emit(self.neu_states)
        if self.common_params.continuous_mode:
            if not self.common_params.draw_layers:
                self.signalNextTact.emit()

    def finishProcessSignals(self):

        self.flag_processing = False

        self.clearLayers()
        if self.common_params.draw_layers:
            self.signalClearVisualize.emit()

        self.signalFinished.emit()

        if self.common_params.processing_type == 'Predict':
            self.io_device.analyzeOutputsPredict(self.common_params.predictStepsNum)
        elif self.common_params.processing_type == 'Novelty filter':
            self.io_device.analyzeOutputsNoveltyFilter(self.common_params.novFiltStepsNum)

        self.io_device.reset()
