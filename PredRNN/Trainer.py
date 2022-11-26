__author__ = 'wuhaixu2016, irccc'

import datetime
import os.path

import torch
import cv2
import PreProcess
import Metrics
import numpy as np


def train(model, ims, real_input_flag, configs, itr):
    '''
    Train a batch of sequence and return mean squared error
    Input:
        model: model which need be trained name
        ims: a batch of sequences which has been reshaped
        real_input_flag: input sequecnces mask list, to display which frame need remenber and need forget
        configs:
            reverse_input: reverse input sequence across timestep to train again, average the loss
            display_interval: how many iterations print information of training
    '''
    cost = model.train(ims, real_input_flag) #loss in this iteration
    if configs.reverse_input: #reverse input batch of sequence
        ims_rev = np.flip(ims, axis=1).copy() #across timestep
        cost += model.train(ims_rev, real_input_flag)
        cost = cost/2

    if itr % configs.display_interval == 0:
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'iteration: ' + str(itr))
        print('training mean squared error: ' + str(cost))


def test(model, test_input_handle, configs, itr):
    '''
    :param model: name of model
    :param test_input_handle: a batch of dataset for testing
    :param configs: test data config details
    :param itr: iteration at this moment
    :return: avg_mse, ssim, psnr, lp
    '''
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'testing')
    test_input_handle.begin(do_shuffle=False)
    res_path = os.path.join(configs.gen_frm_dir, str(itr))
    os.mkdir(res_path)#Create a directory named path
    avg_mse = 0
    batch_id = 0
    img_mse, ssim, psnr = [], [], []
    lp = []

    for i in range(configs.total_length - configs.input_length):
        #initialize in whole predicting process
        img_mse.append(0)
        ssim.append(0)
        psnr.append(0)
        lp.append(0)

    mask_input = configs.input_length

    real_input_flag = np.zeros(
        (configs.batch_size,
         configs.total_length - mask_input - 1,
         configs.img_height // configs.patch_size,
         configs.img_width // configs.patch_size,
         configs.patch_size ** 2 * configs.img_channel))
    #prediction frame are totally treated as input at next iteration

    while(test_input_handle.no_batch_left() == False):
        '''
        iterate every batch
        
            test_ims: groundtruth sequence include input&output sequence
            img_gen: prediction sequence (output sequence)
            
            iterate every output frame
                img_mse: list of every prediction frame mse in whole batch
                avg_mse: sum of all prediction frames mse in whole batch
                lp_loss: the i frame of a batch sequence lp 
                lp: list of every prediction frame lp_loss in whole batch
                psnr: list of every prediction frame psnr in whole batch
                ssim: list of every prediction frame structural similarity in whole batch
            
            save ground truth entire sequence and prediction sequence(output sequence) in NO.1 of a batch
        '''
        batch_id += 1
        test_ims = test_input_handle.get_batch()
        test_dat = PreProcess.reshape_patch(test_ims, configs.patch_size)
        test_ims = test_ims[:, :, :, :, :configs.img_channel]
        img_gen = model.test(test_dat, real_input_flag)
        # prediction frame are totally treated as input at next iteration

        img_gen = PreProcess.reshape_patch_back(img_gen, configs.patch_size)
        output_length = configs.total_length - configs.input_length
        img_out = img_gen[:, -output_length:]

        for i in range(output_length):
            x = test_ims[:, i+configs.input_length, :, :, :]
            gx = img_out[:, i, :, :, :]
            gx = np.maximum(gx, 0)
            gx = np.minimum(gx, 1)
            mse = np.square(x-gx).sum()
            img_mse[i] += mse
            avg_mse += mse

            img_x = np.zeros([configs.batch_size, 3, configs.img_height, configs.img_width])

            if configs.img_channel == 3: #RGB image
                img_x[:, 0, :, :] = x[:, :, :, 0]
                img_x[:, 1, :, :] = x[:, :, :, 1]
                img_x[:, 2, :, :] = x[:, :, :, 2]
            else: #channel is 1 image
                img_x[:, 0, :, :] = x[:, :, :, 0]
                img_x[:, 1, :, :] = x[:, :, :, 0]
                img_x[:, 2, :, :] = x[:, :, :, 0]

            img_x = torch.FloatTensor(img_x)

            img_gx = np.zeros([configs.batch_size, 3, configs.img_height, configs.img_width])

            if configs.img_channel == 3:#RGB image
                img_gx[:, 0, :, :] = gx[:, :, :, 0]
                img_gx[:, 1, :, :] = gx[:, :, :, 1]
                img_gx[:, 2, :, :] = gx[:, :, :, 2]
            else:#channel is 1 image
                img_gx[:, 0, :, :] = gx[:, :, :, 0]
                img_gx[:, 1, :, :] = gx[:, :, :, 0]
                img_gx[:, 2, :, :] = gx[:, :, :, 0]

            img_gx = torch.FloatTensor(img_gx)

            lp_loss = Metrics.loss_fn_alex(img_x, img_gx)
            lp[i] += torch.mean(lp_loss).item()

            real_frm = np.uint8(x * 255)
            pred_frm = np.uint8(gx * 255)

            psnr[i] += Metrics.batch_psnr(pred_frm, real_frm)

            for b in range(configs.batch_size):
                score = Metrics.ssim(pred_frm[b], real_frm[b], full=True, multichannel=True)
                ssim[i] += score

        #save prediction samples
        if batch_id <= configs.num_save_samples:
            path = os.path.join(res_path, str(batch_id))
            os.mkdir(path)
            for i in range(configs.total_length):
                name = 'GroundTruth' + str(i+1) + '.png'
                file_name = os.path.join(path, name)
                img_gt = np.uint8(test_ims[0, i, :, :, :] * 255) #the first batch
                cv2.imwrite(file_name, img_gt)

            for i in range(output_length):
                name = 'Prediction' + str(i+1+configs.input_length) + '.png'
                file_name = os.path.join(path, name)
                img_pd = img_out[0, i, :, :, :] #the first batch
                img_pd = np.maximum(img_pd, 0)
                img_pd = np.minimum(img_pd, 1)
                img_pd = np.uint8(img_pd * 255)
                cv2.imwrite(file_name, img_pd)

        test_input_handle.next()

    '''
    avg_mse: the average of mse every prediction sequence
    ssim: the average list of structural similarity every prediction sequence, include every prediction frame
    psnr: the average list of psnr every batch, include whole batch of sequence in every prediction frame
    lp: the average list of lp every batch, include whole batch of sequence in every prediction frame
    '''
    avg_mse = avg_mse / (batch_id * configs.batch_size) #sum / all batches of sequence
    print('MSE per sequence:' + str(avg_mse))

    for i in range(configs.total_length - configs.input_length):
        print(img_mse[i] / (batch_id * configs.batch_size))#sum / all batches of sequence

    ssim = np.asarray(ssim, dtype=np.float32) / (configs.batch_size * batch_id)#sum / all batches of sequence
    print('Structural Similarity per frame: ' + str(np.mean(ssim)))

    for i in range(configs.total_length - configs.input_length):
        print(ssim[i])

    psnr = np.asarray(psnr, dtype=np.float32) / batch_id
    print('psnr per frame: ' + str(np.mean(psnr)))

    for i in range(configs.total_length - configs.input_length):
        print(psnr[i])

    lp = np.asarray(lp, dtype=np.float32) / batch_id
    print('lpips per frame: ' + str(np.mean(lp)))

    for i in range(configs.total_length - configs.input_length):
        print(lp[i])