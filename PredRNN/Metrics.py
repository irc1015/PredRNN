__author__ = 'irccc'

import numpy as np
from skimage.metrics import structural_similarity
import lpips

def loss_fn_alex(img_x, img_gx):
    loss_fn_alex = lpips.LPIPS(net='alex')
    lp_loss = loss_fn_alex(img_x, img_gx)
    return lp_loss

def batch_psnr(gen_frames, gt_frames):
    if gen_frames.ndim == 3:
        axis = (1, 2)
    elif gen_frames.ndim == 4:
        axis = (1, 2, 3)
    x = np.int32(gen_frames)
    y = np.int32(gt_frames)
    num_pixels = float(np.size(gen_frames[0]))
    #how many pixels in a frame or a batch of frames
    mse = np.sum((x-y)**2, axis=axis, dtype=np.float32) / num_pixels
    psnr = 20 * np.log10(255) - 10 * np.log10(mse)
    return np.mean(psnr)

def ssim(pred_frm, real_frm, full, multichannel):
    score, _ = structural_similarity(pred_frm, real_frm, full=full, multichannel=multichannel)
    return score

