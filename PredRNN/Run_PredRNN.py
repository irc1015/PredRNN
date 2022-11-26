__author__ = 'yunbo, irccc'

import os
import numpy as np
import math
import shutil #high level files operations
import argparse #command-line interfaces
import DataSets_Garden
from Model_Garden_PyTorch import Model
import PreProcess
import Trainer

parser = argparse.ArgumentParser(description='PredRNN version 1')

#GPU / CPU
parser.add_argument('--device', type=str, default='cpu:0')

#train / test
parser.add_argument('--is_training', type=int, default=1)

#dataset setting
parser.add_argument('--dataset_name', type=str, default='mnist')
parser.add_argument('--train_data_paths', type=str, default='data/moving-mnist-example/moving-mnist-train.npz')
parser.add_argument('--valid_data_paths', type=str, default='data/moving-mnist-example/moving-mnist-valid.npz')
parser.add_argument('--save_dir', type=str, default='checkpoints/mnist_predrnn')
parser.add_argument('--gen_frm_dir', type=str, default='results/mnist_predrnn')
parser.add_argument('--input_length', type=int, default=10)
parser.add_argument('--total_length', type=int, default=20)
parser.add_argument('--img_height', type=int, default=64)
parser.add_argument('--img_width', type=int, default=64)
parser.add_argument('--img_channel', type=int, default=1)

#model
parser.add_argument('--model_name', type=str, default='predrnn')
parser.add_argument('--pretrained_model', type=str, default='') #trained model which is used to experiment
parser.add_argument('--num_hidden', type=str, default='64,64,64,64')
parser.add_argument('--filter_size', type=int, default=5)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--patch_size', type=int, default=4)
parser.add_argument('--layer_norm', type=int, default=1)
parser.add_argument('--decouple_beta', type=float, default=0.1)

# scheduled sampling
parser.add_argument('--scheduled_sampling', type=int, default=1)
parser.add_argument('--sampling_stop_iter', type=int, default=50000)
parser.add_argument('--sampling_start_value', type=float, default=1.0)
parser.add_argument('--sampling_changing_rate', type=float, default=0.00002)

# optimization
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--reverse_input', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--max_iterations', type=int, default=80000)
parser.add_argument('--display_interval', type=int, default=100)
parser.add_argument('--test_interval', type=int, default=5000)
parser.add_argument('--snapshot_interval', type=int, default=5000)
parser.add_argument('--num_save_samples', type=int, default=10)
parser.add_argument('--n_gpu', type=int, default=1)

# visualization of memory decoupling
parser.add_argument('--visual', type=int, default=0)
parser.add_argument('--visual_path', type=str, default='./decoupling_visual')

args = parser.parse_args()
print(args)

def schedule_sampling(eta, itr):
    zeros = np.zeros((args.batch_size,
                      args.total_length - args.input_length - 1,
                      args.img_height // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    if not args.scheduled_sampling:
        return 0.0, zeros

    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0
    random_flip = np.random.random_sample((args.batch_size, args.total_length - args.input_length - 1))
    #[batch_size, output_length-1] random list

    true_token = (random_flip < eta)
    #random True/False, at the begining of iterations True much more, at the end True less more

    ones = np.ones((args.img_height // args.patch_size,
                    args.img_width // args.patch_size,
                    args.patch_size ** 2 * args.img_channel))

    zeros = np.zeros((args.img_height // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))

    real_input_flag = []

    for i in range(args.batch_size):
        for j in range(args.total_length - args.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones) #remenber frame
            else:
                real_input_flag.append(zeros)#forget frame

    real_input_flag = np.array(real_input_flag)

    real_input_flag = np.reshape(real_input_flag,
                                 (args.batch_size,
                                  args.total_length - args.input_length - 1,
                                  args.img_height // args.patch_size,
                                  args.img_width // args.patch_size,
                                  args.patch_size ** 2 * args.img_channel))
    return eta, real_input_flag

def train_wrapper(model):
    '''
    train model by all training dataset and save&test model
    :param model: an instance of model
    :return: trained model by iterating
    '''
    if args.pretrained_model:
        model.load(args.pretrained_model)
    #can use pretrained model to train/test

    #get train/test dataset
    train_input_handle, test_input_handle = DataSets_Garden.data_provider(
        args.dataset_name, args.train_data_paths, args.valid_data_paths, args.batch_size,
        args.img_height,args.img_width, seq_length=args.total_length, is_training=True)

    eta = args.sampling_start_value

    '''
    loop in the max iterations
        can run scheduled sampling
        save model every snapshot_interval iteration
        test model every test_interval iteration
    next iteration
    '''
    for itr in range(1, args.max_iterations + 1):
        if train_input_handle.no_batch_left():
            train_input_handle.begin(do_shuffle=True)

        ims = train_input_handle.get_batch()#get a batch of input&output(full) sequences
        ims = PreProcess.reshape_patch(ims, args.patch_size)

        if args.scheduled_sampling == 1:
            eta, real_input_flag = schedule_sampling(eta, itr)

        Trainer.train(model, ims, real_input_flag, args, itr)

        if itr % args.snapshot_interval == 0:
            model.save(itr)

        if itr % args.test_interval == 0:
            Trainer.test(model, test_input_handle, args, itr)

        train_input_handle.next()

def test_wrapper(model):
    model.load(args.pretrained_model)
    test_input_handle = DataSets_Garden.data_provider(args.dataset_name, args.train_data_paths,
                                                      args.valid_data_paths, args.batch_size,
                                                      args.img_height, args.img_width,args.total_length,
                                                      is_training=False
                                                      )
    Trainer.test(model, test_input_handle, args, 'Test Result')

if os.path.exists(args.save_dir):
    shutil.rmtree(args.save_dir) #delete entire directory tree
os.makedirs(args.save_dir)

if os.path.exists(args.gen_frm_dir):
    shutil.rmtree(args.gen_frm_dir)
os.makedirs(args.gen_frm_dir)

print('Initializing PredRNN models')

model = Model(args)

if args.is_training:
    train_wrapper(model)
else:
    test_wrapper(model)