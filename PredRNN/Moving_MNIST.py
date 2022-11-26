__author__ = 'irccc'

import numpy as np
import random

class InputHandle:
    def __init__(self, input_param):
        '''
        Input:
            input_param: a dict which has params of dataset
                paths: the list of dataset paths
                name: name of dataset which for train & test
                input_data_type, minibatch_size, is_output_sequence
        Determine:
            paths, name, input_data_type, output_data_type, minibatch_size, is_output_sequence
            num_paths: how many dataset paths
        Initialize:
            data: DataSet information, include two datasets synthesize into one DateSet
            indices: sequence index
            current_position: the sequence position in indices list
            current_batch_size: the batch_size of current batch, may the last batch_size change
            current_batch_indices: In current batch, sequence index in indices list
            current_input_length:
            current_output_length:
        Run:
            load function
        '''
        self.paths = input_param['paths']
        self.num_paths = len(input_param['paths'])
        self.name = input_param['name']
        self.input_data_type = input_param.get('input_data_type', 'float32')
        #if 'input_data_type' does not exist, return float32
        self.output_data_type = input_param.get('output_data_type', 'float32') # not set in DataSets_Garden.py
        self.minibatch_size = input_param['minibatch_size']
        self.is_output_sequence = input_param['is_output_sequence']
        self.data = {}
        self.indices = []
        self.current_position = 0
        self.current_batch_size = 0
        self.current_batch_indices = []
        self.current_input_length = 0
        self.current_output_length = 0
        self.load()

    def load(self):
        '''
        Load path and make config of dataset
        E.G.
            dat_1: clips shape:(2, 10000, 2)   input_frame_sequence[input_timestep&order, length]
                                               output_frame_sequence[output_timestep&order, length]
                   dims shape:(1,3)  every frame size [1, 64, 64]
                   input_raw_data shape:(200000, 1, 64, 64)  every frame order&size
        '''
        dat_1 = np.load(self.paths[0])
        for key in dat_1.keys():
            self.data[key] = dat_1[key]
        if self.num_paths == 2: #for two DataSet train/test
            dat_2 = np.load(self.paths[1])
            num_clips_1 = dat_1['clips'].shape[1] #the sum of sequence, about 10000
            dat_2['clips'][:,:,0] += num_clips_1
            self.data['clips'] = np.concatenate(dat_1['clips'], dat_2['clips'], axis=1)
            #concatenate input_frame_sequence[input_timestep&order, length] + [input_timestep&order+sequence_sum, length]
            self.data['input_raw_data'] = np.concatenate(dat_1['input_raw_data'], dat_2['input_raw_data'], axis=0)
            self.data['output_raw_data'] = np.concatenate(dat_1['output_raw_data'], dat_2['output_raw_data'], axis=0)
        for key in self.data.keys():
            print(key)
            print(self.data[key].shape)

    def total(self):
        '''
        Return: the sum of sequence
        '''
        return self.data['clips'].shape[1]

    def begin(self, do_shuffle = True):
        '''
        Input:
            do_shuffle: whether shuffle sequence index or not
        Initialize:
            current_position, current_batch_size, current_batch_indices，
            current_input_length， current_output_length
        '''
        self.indices = np.arange(self.total(), dtype='int32')#get squence index list
        if do_shuffle:
            random.shuffle(self.indices)

        self.current_position = 0

        if self.current_position + self.minibatch_size <= self.total():
            self.current_batch_size = self.minibatch_size
        else:
            self.current_batch_size = self.total() - self.current_position #maybe shrink the last batch_size

        self.current_batch_indices = self.indices[self.current_position : self.current_position+self.current_batch_size]
        #In this batch, sequence index in indices list

        self.current_input_length = max(self.data['clips'][0, ind, 1] for ind in self.current_batch_indices)
        self.current_output_length = max(self.data['clips'][1, ind, 1] for ind in self.current_batch_indices)
        #the max length of all of sequences in input/output sequence across current_batch_indices

    def next(self):
        '''
        Adjust the parameter of next batch data
        Update current_position, current_batch_indices, current_input_length, current_output_length
        '''
        self.current_position += self.current_batch_size
        if self.no_batch_left():
            return None
        if self.current_position + self.minibatch_size <= self.total():
            self.current_batch_size = self.minibatch_size
        else:
            self.current_batch_size = self.total() - self.current_position

        self.current_batch_indices = self.indices[self.current_position: self.current_position + self.current_batch_size]
        #

        self.current_input_length = max(self.data['clips'][0, ind, 1] for ind in self.current_batch_indices)
        self.current_output_length = max(self.data['clips'][1, ind, 1] for ind in self.current_batch_indices)


    def no_batch_left(self):
        '''
        Whether no more one batch later this batch
        '''
        if self.current_position >= self.total() - self.current_batch_size:
            return True
        else:
            return False

    def input_batch(self):
        '''
        Make up a whole batch sequence, [length, height, width, channel] every sequence
        '''
        if self.no_batch_left():
            return None
        input_batch = np.zeros((self.current_batch_size, self.current_input_length) + tuple(self.data['dims'][0])).astype(self.input_data_type)
        '''
        [batch, timestep, channel, height, width]
        
        clips shape:(2, 10000, 2)   input_frame_sequence[input_timestep&order, length]
                                    output_frame_sequence[output_timestep&order, length]
        dims shape:(1,3)  every frame size [1, 64, 64]
        input_raw_data shape:(200000, 1, 64, 64)  every frame order&size
        
        data['dims'] --> [[1, 64, 64]]
        data['dims'][0] --> [1, 64, 64]
        '''
        input_batch = np.transpose(input_batch, (0,1,3,4,2))#[batch, timestep, height, width, channel]

        #train a batch_size
        for i in range(self.current_batch_size):
            batch_ind = self.current_batch_indices[i]
            begin = self.data['clips'][0, batch_ind, 0] #input_frame_sequence, the index of sequence, timestep&order
            end = self.data['clips'][0, batch_ind, 0] + self.data['clips'][0, batch_ind, 1]# + this sequence length
            data_slice = self.data['input_raw_data'][begin:end, :, :, :]
            #this sequence frame [timestep, channel, height, width] --> a sequence information
            data_slice = np.transpose(data_slice, (0, 2, 3, 1))#[timestep, height, width, channel]
            input_batch[i, :self.current_input_length, :, :, :] = data_slice
            #[index_in_a_batch_size, sequence_length, height, width, channel]
        input_batch = input_batch.astype(self.input_data_type)
        return input_batch

    def output_batch(self):
        '''
        Make up a whole batch sequence, [length, height, width, channel] every sequence
        '''
        if self.no_batch_left():
            return None
        if (2, 3) == self.data['dims'].shape: #different paths of dataset
            raw_dat = self.data['output_raw_data']
        else:
            raw_dat = self.data['input_raw_data']

        if self.is_output_sequence:
            if (1, 3) == self.data['dims'].shape:
                output_dim = self.data['dims'][0] #[channel, height, width]
            else:
                output_dim = self.data['dims'][1]
            output_batch = np.zeros((self.current_batch_size, self.current_output_length) + tuple(output_dim))
            #[batch, timestep, channel, height, width]
        else:
            output_batch = np.zeros((self.current_batch_size,) + tuple(self.data['dims'][1]))
            #[batch, ]

        #a whole batch_size of output
        for i in range(self.current_batch_size):
            batch_ind = self.current_batch_indices[i]
            begin = self.data['clips'][1, batch_ind, 0]#output_frame_sequence, the index of sequence, timestep&order
            end = self.data['clips'][1, batch_ind, 0] + self.data['clips'][1, batch_ind,1]# + this sequence length
            if self.is_output_sequence:
                data_slice = raw_dat[begin:end, :, :, :]
                #this sequence frame [timestep, channel, height, width] --> a sequence information
                output_batch[i, :data_slice.shape[0], :, :, :] = data_slice
            else:
                data_slice = raw_dat[begin, :, :, :]#[begin_index, channel, height, width] --> a frame information
                output_batch[i, :, :, :] = data_slice

        output_batch = output_batch.astype(self.output_data_type)
        output_batch = np.transpose(output_batch, [0,1,3,4,2])
        #[batch, length, height, width, channel]
        return output_batch

    def get_batch(self):
        '''
        Get a batch input sequence, and a batch output sequence
        make up to a batch full sequence across timestep
        '''
        input_seq = self.input_batch()
        output_seq = self.output_batch()
        batch = np.concatenate((input_seq, output_seq), axis=1) #follow the timestep
        return batch


