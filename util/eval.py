import numpy as np


def evaluate(pred, true):
    mse = np.mean(np.square(pred - true))
    mae = np.mean(np.abs(pred - true))

    return mse, mae


def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()

    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()

    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.4f}MB'.format(all_size))

    return all_size
