__author__ = 'yunbo, irccc'

import numpy as np

def reshape_patch(img_tensor, patch_size):
    '''
    Deal with large image by patch
    param img_tensor: tensor of original image
    param patch_size: divide original image by patch_size

    divide original image by a standard patch, so a image will be divided as
    (img_height // patch_size) * (img_width // patch_size) patches
    And stack all patches across channels -->

    :return: reshape tensor
    '''
    assert 5 == img_tensor.ndim
    #[batch, timestep, height, width, channel]
    batch_size = np.shape(img_tensor)[0]
    seq_length = np.shape(img_tensor)[1]
    img_height = np.shape(img_tensor)[2]
    img_width = np.shape(img_tensor)[3]
    num_channels = np.shape(img_tensor)[4]

    a = np.reshape(img_tensor, [batch_size, seq_length,
                                img_height//patch_size, patch_size,
                                img_width//patch_size, patch_size,
                                num_channels])
    b = np.transpose(a, [0,1,2,3,4,5,6])

    patch_tensor = np.reshape(b, [batch_size, seq_length,
                                  img_height//patch_size, img_width//patch_size,
                                  patch_size*patch_size*num_channels])
    return patch_tensor

def reshape_patch_back(patch_tensor, patch_size):
    '''
    reshape back tensor from patch stack tensor to original image tensor
    '''
    assert 5 == patch_tensor.ndim
    batch_size = np.shape(patch_tensor)[0]
    seq_length = np.shape(patch_tensor)[1]
    patch_height = np.shape(patch_tensor)[2]
    patch_width = np.shape(patch_tensor)[3]
    channels = np.shape(patch_tensor)[4]
    img_channels = channels // (patch_size*patch_size)

    a = np.reshape(patch_tensor, [batch_size, seq_length,
                                  patch_height, patch_width,
                                  patch_size, patch_size,
                                  img_channels])
    b = np.transpose(a, [0,1,2,3,4,5,6])
    img_tensor = np.reshape(b, [batch_size, seq_length,
                                patch_height*patch_size,
                                patch_width*patch_size,
                                img_channels])
    return img_tensor

