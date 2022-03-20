import numpy as np
#from typing import List, Dict
import os

class AbstractIODevice:

    def __init__(self, d: int, q: int):
        self.input_sample = []
        self.output_samples_dict = dict()
        self.iterator = 0
        self.d = d
        self.q = q

        self.reset()

    def getIODeviceState(self):

        output_samples_dict2 = dict()
        for key in self.output_samples_dict.keys():
            output_samples_dict2[key] = np.copy(self.output_samples_dict[key])

        return {'input_sample': np.copy(self.input_sample),
                'output_samples_dict': output_samples_dict2,
                'iterator': self.iterator - 1}

    def setIODeviceState(self, state):
        self.input_sample = state['input_sample']
        self.output_samples_dict = dict()#state['output_samples_dict']
        self.iterator = state['iterator']

    def encode(self, sample):
        pass

    def decode(self, ssp):
        pass

    def getSspFromInput(self):
        pass

    def sendSspToOutput(self, output_field, ssp):
        pass

    def setInputDataFilename(self, filename):
        pass

    def reset(self):
        self.iterator = 0
        self.output_samples_dict = dict()

    def modifyForPredict(self, num):
        self.input_sample = self.input_sample[0:self.iterator*self.d*self.q]
        self.input_sample = np.concatenate((self.input_sample, np.zeros(num*self.d*self.q)))

    def modifyForNoveltyFilter(self, shift, num):
        self.input_sample = self.input_sample[0:min((self.iterator+num)*self.d*self.q, len(self.input_sample))]

# class TestIODevice(AbstractIODevice):
#
#     def __init__(self, d: int, q: int):
#         super(TestIODevice, self).__init__(d, q)
#
#     def encode(self, sample):
#         return np.reshape(sample, (8, 8))
#
#     def decode(self, ssp):
#         return 1.1
#
#     def getSspFromInput(self):
#         if self.iterator >= len(self.input_sample)/64:
#             return []
#         ret = self.encode(self.input_sample[64*self.iterator:64*(self.iterator+1)])
#         self.iterator += 1
#         return ret
#
#     def sendSspToOutput(self, output_field_route_index, ssp):
#         if not output_field_route_index in self.output_samples_dict.keys():
#             self.output_samples_dict[output_field_route_index] = []
#         self.output_samples_dict[output_field_route_index].append(self.decode(ssp))
#         print(self.output_samples_dict[output_field_route_index][len(self.output_samples_dict[output_field_route_index])-1])
#
#     def setInputDataFilename(self, input_data_filename):
#         self.reset()
#         print(input_data_filename)
#         if not os.path.exists(input_data_filename):
#             print('ERROR: file', input_data_filename, ' not exists')
#             return
#         with open(input_data_filename) as f:
#             file_data = f.read()
#             self.input_sample = []
#             for i in range(len(file_data)):
#                 if file_data[i] == "0":
#                     self.input_sample.append(0)
#                 elif file_data[i] == "1":
#                     self.input_sample.append(1)

class WordConnectionsIODevice(AbstractIODevice):

    def __init__(self, d: int, q: int):
        super(WordConnectionsIODevice, self).__init__(d, q)

        dictionary_file = 'example data/connections_dictionary.txt'
        with open(dictionary_file, "r") as dict_file:
            self.dict = dict_file.read().split('\n')


    def encode(self, sample):
        if len(sample) < self.d*self.q:
            print('uncorrect input data')
            return []
        return np.reshape(sample, (self.d, self.q))

    def decode(self, ssp):
        ssp_reshaped = ssp.copy().reshape(self.d * self.q)
        ssp_decoded = ''
        for i in range(len(ssp_reshaped)):
            if ssp_reshaped[i] == -1:
                ssp_decoded += '\n'
                ssp_decoded += self.dict[i]
        return ssp_decoded

    def getSspFromInput(self):
        print('iterator = ', self.iterator)
        if self.iterator >= len(self.input_sample)/(self.d*self.q):
            return []
        ret = self.encode(self.input_sample[self.d*self.q*self.iterator:self.d*self.q*(self.iterator+1)])
        self.iterator += 1
        return ret

    def sendSspToOutput(self, output_field_route_index, ssp):
        if not output_field_route_index in self.output_samples_dict.keys():
            self.output_samples_dict[output_field_route_index] = []
        print('output step', self.iterator, 'in field', output_field_route_index)
        self.output_samples_dict[output_field_route_index].append(self.decode(ssp))
        #print(self.output_samples_dict[output_field_route_index][len(self.output_samples_dict[output_field_route_index])-1])

    def setInputDataFilename(self, input_data_filename):
        self.reset()
        print(input_data_filename)
        if not os.path.exists(input_data_filename):
            print('ERROR: file', input_data_filename, ' not exists')
            return
        with open(input_data_filename) as f:
            file_data = f.read()
            self.input_sample = []
            for i in range(len(file_data)):
                if file_data[i] == "0":
                    self.input_sample.append(0)
                elif file_data[i] == "1":
                    self.input_sample.append(1)

    def analyzeOutputsPredict(self, last_steps_num):
        print(f'Got {last_steps_num} steps')

    def analyzeOutputsNoveltyFilter(self, last_steps_analyzed):
        print('analyzing novelty filter...')
        for field_id in self.output_samples_dict.keys():
            len0 = len(self.output_samples_dict[field_id])
            if len0 < last_steps_analyzed or not len0:
                print(f'in field {field_id} not enough output data samples for analyzing')
                continue
            field_last_steps = self.output_samples_dict[field_id][len0-last_steps_analyzed:len0]
            for i in range(len(field_last_steps)):
                print(f'\n\nfield {field_id} step {i} results:')
                print(field_last_steps[i])
                print('\n')
