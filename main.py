import argparse
from time import time

import torch

from solver.solver import Solver
from util.seed import fixSeed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    """ Basic """
    parser.add_argument('--model', type=str, default='DLinear',
                        choices=['Informer', 'DLinear', 'TimesNet', 'iTransformer'], help='backbone network')
    parser.add_argument('--flag', type=str, default='Standard',
                        choices=['Plugin', 'Standard'], help='GLAFF or Standard')
    parser.add_argument('--only_test', default=False, action='store_true', help='only test the model')

    """ Data """
    parser.add_argument('--data_path', type=str, default='./dataset/', help='path to dataset')
    parser.add_argument('--save_path', type=str, default='./checkpoint/', help='path to save model')
    parser.add_argument('--dataset', type=str, default='Traffic', help='dataset name')
    parser.add_argument('--hist_len', type=int, default=96, help='length of history window')
    parser.add_argument('--pred_len', type=int, default=192, help='length of prediction window')

    """ Plugin """
    parser.add_argument('--dim', type=int, default=256, help='dimension of hidden state')
    parser.add_argument('--dff', type=int, default=512, help='dimension of feed forward')
    parser.add_argument('--head_num', type=int, default=8, help='number of heads')
    parser.add_argument('--layer_num', type=int, default=2, help='number of layers')
    parser.add_argument('--dropout', type=float, default=0.6, help='dropout rate')
    parser.add_argument('--q', type=float, default=0.75, help='quantile')

    """ Optim """
    parser.add_argument('--itr', type=int, default=1, help='number of iterations')
    parser.add_argument('--epoch', type=int, default=10, help='number of epochs')
    parser.add_argument('--patience', type=int, default=3, help='patience for early stopping')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')

    """ GPU """
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('--device', type=int, default=0, help='device id')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    print('\n=====================Args========================')
    print(args)
    print('=================================================\n')

    fixSeed(args.seed)

    for ii in range(args.itr):
        setting = '{0}_{1}_{2}_{3}_{4}_{5}'.format(
            args.model,
            args.flag,
            args.dataset,
            args.hist_len,
            args.pred_len,
            ii
        )

        print('\n>>>>>>>>  initing : {}  <<<<<<<<\n'.format(setting))
        solver = Solver(args, setting)

        if not args.only_test:
            print('\n>>>>>>>>  training : {}  <<<<<<<<\n'.format(setting))
            start = time()
            epoch = solver.train()
            train_time = (time() - start) / epoch
            print('Training Time: {:.4f}s'.format(train_time))

        print('\n>>>>>>>>  testing : {}  <<<<<<<<\n'.format(setting))
        start = time()
        res = solver.test()
        test_time = time() - start
        print('Testing Time: {:.4f}s'.format(test_time))

        f = open('./result.txt', 'a')
        f.write(setting + "  \n")
        if not args.only_test:
            f.write('Train:{0:.4f} s, Test:{1:.4f} s, Size:{2:.4f} MB\n'.format(train_time, test_time, res['size']))
        f.write('MSE:{0:.4f}, MAE:{1:.4f}\n'.format(res['MSE'], res['MAE']))
        f.write('\n')
        f.close()

        torch.cuda.empty_cache()

    print('\nDone!')
