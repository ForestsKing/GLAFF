from einops import rearrange
from sklearn.metrics import mean_squared_error, mean_absolute_error


def evaluate(pred, true):
    pred = rearrange(pred, 'N L C -> (N C) L')
    true = rearrange(true, 'N L C -> (N C) L')

    res = {
        'MSE': mean_squared_error(true, pred),
        'MAE': mean_absolute_error(true, pred),
    }

    return res


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
