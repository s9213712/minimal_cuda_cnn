"""Weight checkpointing and device pointer cleanup."""

import numpy as np

from cuda_backend import g2h, gpu_zeros, lib, upload
from train_config import C1_IN, C1_OUT, C2_IN, C2_OUT, C3_IN, C3_OUT, C4_IN, C4_OUT, FC_IN, KH, KW


def upload_weights(w_conv1, w_conv2, w_conv3, w_conv4, fc_w, fc_b):
    return (
        upload(w_conv1),
        upload(w_conv2),
        upload(w_conv3),
        upload(w_conv4),
        upload(fc_w),
        upload(fc_b),
    )


def init_velocity_buffers():
    return (
        gpu_zeros(C1_OUT * C1_IN * KH * KW),
        gpu_zeros(C2_OUT * C2_IN * KH * KW),
        gpu_zeros(C3_OUT * C3_IN * KH * KW),
        gpu_zeros(C4_OUT * C4_IN * KH * KW),
        gpu_zeros(10 * FC_IN),
        gpu_zeros(10),
    )


def save_checkpoint(path, epoch, val_acc, lr_conv1, lr_conv, lr_fc, device_weights):
    d_w_conv1, d_w_conv2, d_w_conv3, d_w_conv4, d_fc_w, d_fc_b = device_weights
    np.savez(
        path,
        epoch=np.int32(epoch),
        val_acc=np.float32(val_acc),
        lr_conv1=np.float32(lr_conv1),
        lr_conv=np.float32(lr_conv),
        lr_fc=np.float32(lr_fc),
        w_conv1=g2h(d_w_conv1, C1_OUT * C1_IN * KH * KW),
        w_conv2=g2h(d_w_conv2, C2_OUT * C2_IN * KH * KW),
        w_conv3=g2h(d_w_conv3, C3_OUT * C3_IN * KH * KW),
        w_conv4=g2h(d_w_conv4, C4_OUT * C4_IN * KH * KW),
        fc_w=g2h(d_fc_w, 10 * FC_IN),
        fc_b=g2h(d_fc_b, 10),
    )


def reload_weights_from_checkpoint(path, device_weights):
    free_weights(device_weights)
    ckpt = np.load(path)
    fc_w = ckpt["fc_w"].astype(np.float32)
    fc_b = ckpt["fc_b"].astype(np.float32)
    new_device_weights = (
        upload(ckpt["w_conv1"].astype(np.float32)),
        upload(ckpt["w_conv2"].astype(np.float32)),
        upload(ckpt["w_conv3"].astype(np.float32)),
        upload(ckpt["w_conv4"].astype(np.float32)),
        upload(fc_w),
        upload(fc_b),
    )
    return ckpt, fc_w, fc_b, new_device_weights


def free_weights(device_weights):
    for ptr in device_weights:
        lib.gpu_free(ptr)
