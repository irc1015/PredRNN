__author__ = 'yunbo, irccc(alter code as same as paper)'

import torch
import torch.nn as nn

class STLSTMCell(nn.Module):
    '''
    Input: frame, H, C, M
    Output: H, C, M
    '''
    def __init__(self, in_channel, height, width, num_hidden, filter_size, stride, layer_norm):
        '''
        layer_norm: whether model uses the Batchnormalization
        '''
        super(STLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0

        if layer_norm:
            '''
            conv_x: Convolution layer of Frame X
            conv_h: Convolution layer of Hidden State H 
            conv_m: Convolution layer of Spatiotemporal Memory M
            conv_o: Convolution layer of Output O
            [channel, height, width]
            '''
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channels=in_channel,
                          out_channels=num_hidden*7,
                          kernel_size=filter_size,
                          stride=stride,
                          padding=self.padding), # padding=0 --> valid; padding=1 --> same
                nn.LayerNorm( [num_hidden*7, height, width] )
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(in_channels=num_hidden,
                          out_channels=num_hidden * 4,
                          kernel_size=filter_size,
                          stride=stride,
                          padding=self.padding),
                nn.LayerNorm( [num_hidden * 4, height, width] )
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(in_channels=num_hidden,
                          out_channels=num_hidden * 3,
                          kernel_size=filter_size,
                          stride=stride,
                          padding=self.padding),
                nn.LayerNorm([num_hidden * 3, height, width])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(in_channels=num_hidden * 2,
                          out_channels=num_hidden,
                          kernel_size=filter_size,
                          stride=stride,
                          padding=self.padding),
                nn.LayerNorm( [num_hidden, height, width] )
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channels=in_channel,
                          out_channels=num_hidden * 7,
                          kernel_size=filter_size,
                          stride=stride,
                          padding=self.padding),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(in_channels=num_hidden,
                          out_channels=num_hidden * 4,
                          kernel_size=filter_size,
                          stride=stride,
                          padding=self.padding),
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(in_channels=num_hidden,
                          out_channels=num_hidden * 3,
                          kernel_size=filter_size,
                          stride=stride,
                          padding=self.padding),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(in_channels=num_hidden*2,
                          out_channels=num_hidden,
                          kernel_size=filter_size,
                          stride=stride,
                          padding=self.padding),
            )
        self.conv_last = nn.Conv2d(in_channels=num_hidden * 2,
                                   out_channels=num_hidden,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0)

    def forward(self, x_t, h_t, c_t, m_t):
        '''
        x_t: Frame X at timestep t
        h_t: Hidden State H at timestep t
        c_t: Standard Temporal Cell C at timestep t
        m_t: Spatiotemporal Memory M at timestep t
        [batch, channel, height, width] --> about patch_size

        x_concat shape is torch.Size([8, 896, 16, 16])
        h_concat shape is torch.Size([8, 512, 16, 16])
        m_concat shape is torch.Size([8, 384, 16, 16])
        '''
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)


        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        '''
        i_x: Input Gate for Frame X at timesetp t
        f_x: Forget Gate for Frame X at timesetp t
        g_x: Input Modulation Gate for Frame X at timesetp t
        i_x_prime: Input Gate for Frame X and Spatiotemporal Memory Flow at timesetp t
        f_x_prime: Forget Gate for Frame X and Spatiotemporal Memory Flow at timesetp t
        g_x_prime: Input Modulation Gate for Frame X and Spatiotemporal Memory Flow at timesetp t
        o_x: Output Gate for Frame X at timesetp t
        Split the output of conv_x across height
        '''

        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        '''
        i_h: Input Gate for Hidden State H at timesetp t-1
        f_h: Forget Gate for Hidden State H at timesetp t-1
        g_h: Input Modulation Gate for Hidden State H at timesetp t-1
        o_h: Output Gate for Hidden State H at timesetp t-1
        Split the output of conv_h across height
        '''

        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)
        '''
        i_m: Input Gate for Spatiotemporal Memory M at timesetp t-1
        f_m: Forget Gate for Spatiotemporal Memory M at timesetp t-1
        g_m: Input Modulation Gate for Spatiotemporal Memory M at timesetp t-1
        Split the output of conv_h across height
        '''

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)

        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new
