__author__ = 'irccc'

import os
import torch
from torch.optim import Adam
import PredRNN_Model_Pytorch as PredRNN

class Model(object):
    def __init__(self, configs):
        '''
        Input:
            configs:
                configs.num_hidden -->  num_hidden: the list of nodes of every layer
                                        num_layers: the length of num_hidden list
                configs.model_name --> the name of model is to match
        Output: a Network which set successful
                an Adam optimizer
        '''
        self.configs = configs
        self.num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        self.num_layers = len(self.num_hidden)
        networks = {
            'predrnn':PredRNN.RNN,
        }
        if configs.model_name in networks:
            Network = networks[configs.model_name]
            self.network = Network(self.num_layers, self.num_hidden, configs).to(configs.device)
            #change the existing tensors, so use to() method
        else:
            raise ValueError('Unknown Network Name: {}'.format(configs.model_name))

        self.optimizer = Adam(self.network.parameters(),lr=configs.lr)

    def save(self, itr):
        '''
        Input:
            itr: current epoch
        Output:
            checkpoint at current epoch in specify path
        '''
        stats = {} #NULL dict
        stats['net_param'] = self.network.state_dict()
        checkpoint_path = os.path.join(self.configs.save_dir, str(itr)+'_Model.pt')
        torch.save(stats, checkpoint_path)
        print('Saved model to {} '.format(checkpoint_path))

    def load(self, checkpoint_path):
        stats = torch.load(checkpoint_path) #state dict is loaded
        self.network.load_state_dict(stats['net_param'])#the state then restored
        print('Loaded Model from {}'.format(checkpoint_path))

    def train(self, frames, mask):
        '''
        Input:
            frames:
            mask:
        Output: loss
        '''
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        self.optimizer.zero_grad()#Sets the gradients of all optimized to zero.
        next_frames, loss = self.network(frames_tensor, mask_tensor)
        loss.backward()
        self.optimizer.step()#Performs a single optimization step (parameter update)
        return loss.detach().cpu().numpy()#need another tensor will never gradient

    def test(self, frames, mask):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        next_frames, _ = self.network(frames_tensor, mask_tensor) #need a unuseful variable to get the loss
        return next_frames.detach().cpu().numpy()  # need another tensor will never gradient