import configparser
import os
import json
import numpy as np

class CommonParams:
    def __init__(self, file_path):
        self.file_path = file_path
        if not os.path.exists(self.file_path):
            raise 'ini file not found'

        self.windowState, self.geometry = bytes(), bytes()
        try:
            with open(os.path.join(os.getcwd(), 'GUI', 'windowState'), "rb") as windowState_file:
                self.windowState = windowState_file.read()
            with open(os.path.join(os.getcwd(), 'GUI', 'geometry'), "rb") as geometry_file:
                self.geometry = geometry_file.read()
        except:
            print('reading window state error')

        self.config = configparser.ConfigParser()
        self.config.read(self.file_path)

        self.processing_type = self.config['MainParams'].get('processing_type', 'Predict')
        self.predictStepsNum = self.config['Forecasting'].getint('predictStepsNum', 0)
        self.novFiltWeightsGain = self.config['NoveltyFiltering'].getfloat('novFiltWeightsGain', 0.0)
        self.novFiltDetectBorder = self.config['NoveltyFiltering'].getfloat('novFiltDetectBorder', 0.0)
        self.novFiltStepsNum = self.config['NoveltyFiltering'].getint('novFiltStepsNum', 0)
        self.initHistoryPeriod = self.config['NoveltyFiltering'].getint('initHistoryPeriod', 5)
        self.rnn2DelayNumTacts = self.config['NoveltyFiltering'].getint('rnn2DelayNumTacts', 2)

        self.tab_rnn1_index = self.config['GuiParams'].getint('tab_rnn1_index', 0)
        self.tab_rnn2_index = self.config['GuiParams'].getint('tab_rnn2_index', 0)

        self.continuous_mode = self.config['MainParams'].getboolean('continuous_mode', False)

        self.draw_layers = self.config['MainParams'].getboolean('draw_layers', True)

        self.L = self.config['RnnGeometry'].getint('L', 5)
        self.M = self.config['RnnGeometry'].getint('M', 5)
        self.d = self.config['RnnGeometry'].getint('d', 5)
        self.q = self.config['RnnGeometry'].getint('q', 5)

        self.data_block_size = self.d*self.q

        self.ssp_interval = self.config['RnnDataStreaming'].getint('SspSubmitInterval', 6)
        self.refract_interval = self.config['NeuronParams'].getint('RefractInterval', 5)
        limit_coeff = self.config['NeuronParams'].getint('limitCoeff', 1000)

        self.neuron_current_value_limit = np.float64(self.L * self.M * self.d * self.q * limit_coeff)

        if self.L <= 0 or self.M <= 0 or self.d <= 0 or self.q <= 0 or self.ssp_interval <= 0 or self.ssp_interval % 2 != 0:
            raise 'Uncorrect Ini value'

        self.route_type = self.config['RnnGeometry'].getint('route_type', 0)

    def rewrite(self):

        try:
            with open(os.path.join(os.getcwd(), 'GUI', 'windowState'), "wb+") as windowState_file:
                windowState_file.write(self.windowState)
            with open(os.path.join(os.getcwd(), 'GUI', 'geometry'), "wb+") as geometry_file:
                geometry_file.write(self.geometry)
        except:
            print('writing window state error')

        self.config.set('MainParams', 'processing_type', str(self.processing_type))
        self.config.set('Forecasting', 'predictStepsNum', str(self.predictStepsNum))
        self.config.set('NoveltyFiltering', 'novFiltWeightsGain', str(self.novFiltWeightsGain))
        self.config.set('NoveltyFiltering', 'novFiltDetectBorder', str(self.novFiltDetectBorder))
        self.config.set('NoveltyFiltering', 'novFiltStepsNum', str(self.novFiltStepsNum))

        self.config.set('GuiParams', 'tab_rnn1_index', str(self.tab_rnn1_index))
        self.config.set('GuiParams', 'tab_rnn2_index', str(self.tab_rnn2_index))

        with open(self.file_path, "w") as config_file:
            self.config.write(config_file)


class RnnParams:
    def __init__(self, file_path: str, common_params):
        self.file_path = file_path
        if not os.path.exists(self.file_path):
            raise 'ini file not found'

        self.config = configparser.ConfigParser()
        self.config.read(self.file_path)

        self.alpha = self.config['SynapsesParams'].getfloat('alpha', 20)
        self.h = self.config['SynapsesParams'].getfloat('h', 2)

        self.input_data_filename = self.config['IOParams'].get('input_data_filename', '')

        self.border_type = self.config['NeuronParams'].get('border_type', 'Const')
        self.border_Const_value = self.config['NeuronParams'].getfloat('border_Const_value', 1.0)
        self.border_Concurrent_winners = self.config['NeuronParams'].getint('border_Concurrent_winners', 1)

        output_fields_dict = self.config.items("OutputFields")
        if not len(output_fields_dict):
            self.output_fields = [{'Lr': 1, 'L': common_params.L - 1, 'M': common_params.M - 1}]
        else:
            self.output_fields = []
            for item in output_fields_dict:
                output_field = json.loads(item[1])
                if len(output_field) != 3:  # Lr, L, M
                    print('uncorrect output field format')
                    continue
                if output_field[0] != 0 and output_field[0] != 1 or output_field[1] >= common_params.L or \
                        output_field[1] < 0 or output_field[2] >= common_params.M or output_field[2] < 0:
                    print('uncorrect output field format')
                    continue
                self.output_fields.append({'Lr': output_field[0], 'L': output_field[1], 'M': output_field[2]})

    def rewrite(self):

        self.config.set('SynapsesParams', 'alpha', str(self.alpha))
        self.config.set('SynapsesParams', 'h', str(self.h))

        self.config.set('IOParams', 'input_data_filename', str(self.input_data_filename))

        self.config.set('NeuronParams', 'border_type', str(self.border_type))
        self.config.set('NeuronParams', 'border_Const_value', str(self.border_Const_value))
        self.config.set('NeuronParams', 'border_Concurrent_winners', str(self.border_Concurrent_winners))

        with open(self.file_path, "w") as config_file:
            self.config.write(config_file)


class Rnn1Params(RnnParams):
    def __init__(self, file_path: str, common_params):
        super(Rnn1Params, self).__init__(file_path, common_params)

        self.gDec = self.config['SynapsesParams'].getfloat('gDec', 0.001)
        self.gInc = self.config['SynapsesParams'].getfloat('gInc', 0.1)
        self.gSum = self.config['SynapsesParams'].getint('gSum', -1)
        self.gamma = self.config['SynapsesParams'].getfloat('gamma', 0.5)

        self.flag_clear_learning = self.config['ControlParams'].getboolean('flag_clear_learning', True)
        self.flag_learning = self.config['ControlParams'].getboolean('flag_learning', True)

    def rewrite(self):
        super(Rnn1Params, self).rewrite()

        self.config.set('SynapsesParams', 'gInc', str(self.gInc))
        self.config.set('SynapsesParams', 'gDec', str(self.gDec))

        self.config.set('ControlParams', 'flag_clear_learning', str(self.flag_clear_learning))
        self.config.set('ControlParams', 'flag_learning', str(self.flag_learning))

        with open(self.file_path, "w") as config_file:
            self.config.write(config_file)


class Rnn2Params(RnnParams):
    def __init__(self, file_path: str, common_params):
        super(Rnn2Params, self).__init__(file_path, common_params)

        self.flag_clear_learning = False