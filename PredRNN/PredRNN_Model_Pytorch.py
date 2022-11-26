__author__ = 'yunbo, irccc'

import torch
import torch.nn as nn
from STLSTM_PyTorch import STLSTMCell

class RNN(nn.Module):
    '''
    Input:
        num_layers: the number of layers
        num_hidden: the list of nodes of every layer
        configs:
            patch_size, img_channel, img_width, img_height  --> frame_channel, width(patch number), height(patch number)

        frame_tensor: the frame squence
        mask_true:

    Output:
        next_frame: the list of next frame of a batch size of sequences from the first prediction frame
        loss: the entirety mean squared error
    '''
    def __init__(self, num_layers, num_hidden, configs):
        super(RNN, self).__init__()

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []
        '''
        cell_list: STLSTMCell
        width: How many patches in img_width 
        height: How many patches in img_height 
        '''
        width = configs.img_width // configs.patch_size
        height = configs.img_height // configs.patch_size

        self.MSE_criterion = nn.MSELoss() #mean squared error

        for i in range(num_layers):
            in_channel = self.frame_channel if i==0 else num_hidden[i-1]
            cell_list.append(
                STLSTMCell(in_channel=in_channel,
                           height=height, width=width,
                           num_hidden=num_hidden[i],
                           filter_size=configs.filter_size,
                           stride=configs.stride,
                           layer_norm=configs.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list) #Holds every STLSTMCell in a list.

        self.conv_last = nn.Conv2d(in_channels=num_hidden[num_layers-1],
                                   out_channels=self.frame_channel,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   bias=False) #Make the next frame smooth

    def forward(self, frames_tensor, mask_true):
            # [batch, timestep, height, width, channel] -> [batch, timestep, channel, height, width]
            frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
            mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

            batch = frames.shape[0]
            height = frames.shape[3]
            width = frames.shape[4]

            '''
            next_frames:
            h_t: Hidden State H at timestep t
            c_t: Standard Temporal Cell C at timestep t
            '''
            next_frames = []
            h_t = []
            c_t = []
            for i in range(self.num_layers):
                zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
                # change the existing tensors, so use to() method
                h_t.append(zeros)
                c_t.append(zeros)
                #initial h_t, c_t

            memory = torch.zeros([batch, self.num_hidden[0], height, width]).to(self.configs.device)
            # change the existing tensors, so use to() method

            for t in range(self.configs.total_length - 1): #19 frames
                #total_length: a whole frames sequence
                #until predict the last frame to make up a total_length sequence
                if t < self.configs.input_length: #0-9
                    net = frames[:, t]
                else:
                    net = mask_true[:, t-self.configs.input_length] * frames[:, t] + (1 - mask_true[:, t-self.configs.input_length]) * x_gen
                    '''merge a frame which has been remenbered and a  predicting frame which has been forgotten
                        weight of prediction change more and more important as iteration bigger '''
                h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)

                for i in range(1, self.num_layers):
                    h_t[i], c_t[i], memory = self.cell_list[i](h_t[i-1], h_t[i], c_t[i], memory)
                    #h_t[i-1] substitute for input frame in other layers

                x_gen = self.conv_last(h_t[self.num_layers - 1])
                next_frames.append(x_gen)

            next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
            #[timestep, batch, channel, height, width] -> [batch, timestep, height, width, channel]
            loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
            return next_frames, loss
