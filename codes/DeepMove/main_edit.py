# coding: utf-8


from math import inf
from numpy.core.numeric import Inf
from model_edit import TrajPreSimple1, TrajPreSimple2, TrajPreAttnAvgLongUser, TrajPreLocalAttnLong
from train_edit import RnnParameterData, markov, generate_input_history, train_simple, generate_input_list, \
    generate_input_long_history2, generate_input_long_history

# import torch
# import torch.nn as nn
# import torch.optim as optim

import os
import json
import time
import argparse
import pickle
import numpy as np
from json import encoder
import random

encoder.FLOAT_REPR = lambda o: format(o, '.3f')


def run(args):
    parameters = RnnParameterData(loc_emb_size=args.loc_emb_size, uid_emb_size=args.uid_emb_size,
                                  voc_emb_size=args.voc_emb_size, tim_emb_size=args.tim_emb_size,
                                  hidden_size=args.hidden_size, dropout_p=args.dropout_p,
                                  data_name=args.data_name, lr=args.learning_rate,
                                  lr_step=args.lr_step, lr_decay=args.lr_decay, L2=args.L2, rnn_type=args.rnn_type,
                                  optim=args.optim, attn_type=args.attn_type,
                                  clip=args.clip, epoch_max=args.epoch_max, history_mode=args.history_mode,
                                  model_mode=args.model_mode, data_path=args.data_path, save_path=args.save_path)

    argv = {'loc_emb_size': args.loc_emb_size, 'uid_emb_size': args.uid_emb_size, 'voc_emb_size': args.voc_emb_size,
            'tim_emb_size': args.tim_emb_size, 'hidden_size': args.hidden_size,
            'dropout_p': args.dropout_p, 'data_name': args.data_name, 'learning_rate': args.learning_rate,
            'lr_step': args.lr_step, 'lr_decay': args.lr_decay, 'L2': args.L2, 'act_type': 'selu',
            'optim': args.optim, 'attn_type': args.attn_type, 'clip': args.clip, 'rnn_type': args.rnn_type,
            'epoch_max': args.epoch_max, 'history_mode': args.history_mode, 'model_mode': args.model_mode}

    print('*' * 15 + 'start training' + '*' * 15)
    print('model_mode:{} history_mode:{} users:{}'.format(
        parameters.model_mode, parameters.history_mode, parameters.uid_size))

    if parameters.model_mode in ['simple', 'simple_long']:
        model = TrajPreSimple1(hidden_dim=parameters.hidden_size,
                               loc_dim=parameters.loc_size, tim_dim=parameters.tim_size)
    elif parameters.model_mode == 'attn_avg_long_user':
        model = TrajPreAttnAvgLongUser(parameters=parameters)
    elif parameters.model_mode == 'attn_local_long':
        model = TrajPreLocalAttnLong(parameters=parameters)
    if args.pretrain == 1:
        # model.load_state_dict(torch.load(
        #     "../pretrain/" + args.model_mode + "/res.m", map_location=torch.device('cpu')))
        pass

    # if 'max' in parameters.model_mode:
    #     parameters.history_mode = 'max'
    # elif 'avg' in parameters.model_mode:
    #     parameters.history_mode = 'avg'
    # else:
    #     parameters.history_mode = 'whole'

    # criterion = nn.NLLLoss()
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=parameters.lr,
    #                        weight_decay=parameters.L2)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=parameters.lr_step,
    #                                                  factor=parameters.lr_decay, threshold=1e-3)

    metrics = {'train_loss': [], 'valid_loss': [],
               'valid_accuracy': [], 'valid_acc': {},
               'loc': [], 'tim': [], 'test_loc': [], 'test_tim': [],
               'train_target': [], 'test_target': []
               }

    candidate = parameters.data_neural.keys()

    # print(candidate)
    avg_acc_markov, users_acc_markov = markov(parameters, candidate)
    metrics['markov_acc'] = users_acc_markov

    if 'long' in parameters.model_mode:
        long_history = True
    else:
        long_history = False

    if long_history is False:
        data_train, train_idx = generate_input_history(parameters.data_neural, 'train', mode2=parameters.history_mode,
                                                       candidate=candidate)
        data_test, test_idx = generate_input_history(parameters.data_neural, 'test', mode2=parameters.history_mode,
                                                     candidate=candidate)

        metrics['loc'], metrics['tim'], metrics['train_target'] = generate_input_list(
            'train', train_idx, data_train, parameters.model_mode)
        metrics['test_loc'], metrics['test_tim'], metrics['test_target'] = generate_input_list(
            'test', test_idx, data_test, parameters.model_mode)

    elif long_history is True:
        if parameters.model_mode == 'simple_long':
            data_train, train_idx = generate_input_long_history2(
                parameters.data_neural, 'train', candidate=candidate)
            data_test, test_idx = generate_input_long_history2(
                parameters.data_neural, 'test', candidate=candidate)

        else:
            data_train, train_idx = generate_input_long_history(
                parameters.data_neural, 'train', candidate=candidate)
            data_test, test_idx = generate_input_long_history(
                parameters.data_neural, 'test', candidate=candidate)

    print('users:{} markov:{} train:{} test:{}'.format(len(candidate), avg_acc_markov,
                                                       len([
                                                           y for x in train_idx for y in train_idx[x]]),
                                                       len([y for x in test_idx for y in test_idx[x]])))
    SAVE_PATH = args.save_path
    tmp_path = 'mycheckpoint/'
    os.makedirs(SAVE_PATH + tmp_path, exist_ok=True)
#    os.mkdir(SAVE_PATH + tmp_path)

    loc_len = len(metrics['test_loc'])
    divide_len = int(loc_len * 0.25)
    lr = parameters.lr
    lr_step = parameters.lr_step
    lr_decay = parameters.lr_decay
    prev_loss = [inf] * lr_step
    # isValidation = bool()
    # train and valiadation
    # test_loss, test_acc = train_simple(
    #     model, metrics['test_loc'], metrics['test_tim'], metrics['test_target'], lr, True)
    # print('==>Test Acc:{:.4f} Loss:{:.4f}'.format(
    #     test_acc, test_loss))
    for epoch in range(parameters.epoch):

        random.seed(epoch)

        st = time.time()
        if args.pretrain == 1:
            break
        elif args.pretrain == 0:

            if parameters.model_mode == 'simple':

                cc = list(
                    zip(metrics['loc'], metrics['tim'], metrics['train_target']))
                random.shuffle(cc)
                loc_list, tim_list, target = zip(*cc)

                training_loc_list, validation_loc_list = loc_list[:divide_len], loc_list[divide_len:]
                training_tim_list, validation_tim_list = tim_list[:divide_len], tim_list[divide_len:]
                training_target, validation_target = target[:divide_len], target[divide_len:]
                # training_loc_list, validation_loc_list = loc_list, loc_list
                # training_tim_list, validation_tim_list = tim_list, tim_list
                # training_target, validation_target = target, target

                model, avg_loss = train_simple(
                    model, training_loc_list, training_tim_list, training_target, lr, False)
                print('==>Train Epoch:{:0>2d} Loss:{:.4f} lr:{:.5f}'.format(
                    epoch, avg_loss, lr))
                metrics['train_loss'].append(avg_loss)

                vavg_loss, vavg_acc = train_simple(
                    model, validation_loc_list, validation_tim_list, validation_target, lr, True)
                print('==>Validation Acc:{:.4f} Loss:{:.4f}'.format(
                    vavg_acc, vavg_loss))

                metrics['valid_loss'].append(vavg_loss)
                metrics['valid_accuracy'].append(vavg_acc)
                # metrics['valid_acc'][epoch] = users_acc

                save_name_tmp = 'ep_' + str(epoch) + '.pkl'

                with open(SAVE_PATH + tmp_path + save_name_tmp, 'wb') as f:
                    as_str = pickle.dumps(model)
                    f.write(as_str)
                    f.close()

            # static one
                # if epoch % lr_step == 0 and epoch > 0:
            # dynamic one

                if avg_loss > max(prev_loss):
                    # load_epoch = np.argmax(metrics['valid_accuracy'])
                    parameters.lr *= lr_decay
                    lr = parameters.lr
                    last = np.argmax(metrics['valid_accuracy'][::-1])
                    load_epoch = len(metrics['valid_accuracy']) - last - 1
                    load_name_tmp = 'ep_' + str(load_epoch) + '.pkl'
                    with open(SAVE_PATH + tmp_path + load_name_tmp, 'rb') as f:
                        model = pickle.loads(f.read())
                    # model.load_state_dict(torch.load(
                    #     SAVE_PATH + tmp_path + load_name_tmp))
                    print('load epoch={} model state'.format(load_epoch))

            prev_loss[: -1], prev_loss[-1] = prev_loss[1:], avg_loss

            if epoch == 0:
                print('single epoch time cost:{}'.format(time.time() - st))
            if lr <= 0.9 * 1e-5:
                break

    last = np.argmax(metrics['valid_accuracy'][::-1])
    mid = len(metrics['valid_accuracy']) - last - 1
    # print(mid)
    load_name_tmp = 'ep_' + str(mid) + '.pkl'

    with open(SAVE_PATH + tmp_path + load_name_tmp, 'rb') as f:
        model = pickle.loads(f.read())
    # test
    test_loss, test_acc = train_simple(
        model, metrics['test_loc'], metrics['test_tim'], metrics['test_target'], lr, True)
    print('==>Test Acc:{:.4f} Loss:{:.4f}'.format(
        test_acc, test_loss))

    save_name = 'res'
    # json.dump({'args': argv, 'metrics': metrics}, fp=open(
    #     SAVE_PATH + save_name + '.rs', 'w'), indent=4)
    metrics_view = {'train_loss': [], 'valid_loss': [], 'valid_accuracy': []}
    for key in metrics_view:
        metrics_view[key] = metrics[key]
    json.dump({'args': argv, 'metrics': metrics_view}, fp=open(
        SAVE_PATH + save_name + '.txt', 'w'), indent=4)

    # save model to res.pkl
    with open(SAVE_PATH + tmp_path + save_name_tmp, 'wb') as f:
        as_str = pickle.dumps(model)
        f.write(as_str)
        f.close()

    for rt, dirs, files in os.walk(SAVE_PATH + tmp_path):
        for name in files:
            remove_path = os.path.join(rt, name)
            os.remove(remove_path)
    os.rmdir(SAVE_PATH + tmp_path)

    return


def load_pretrained_model(config):
    res = json.load(open("../pretrain/" + config.model_mode + "/res.txt"))
    args = Settings(config, res["args"])
    return args


class Settings(object):
    def __init__(self, config, res):
        self.data_path = config.data_path
        self.save_path = config.save_path
        self.data_name = res["data_name"]
        self.epoch_max = res["epoch_max"]
        self.learning_rate = res["learning_rate"]
        self.lr_step = res["lr_step"]
        self.lr_decay = res["lr_decay"]
        self.clip = res["clip"]
        self.dropout_p = res["dropout_p"]
        self.rnn_type = res["rnn_type"]
        self.attn_type = res["attn_type"]
        self.L2 = res["L2"]
        self.history_mode = res["history_mode"]
        self.model_mode = res["model_mode"]
        self.optim = res["optim"]
        self.hidden_size = res["hidden_size"]
        self.tim_emb_size = res["tim_emb_size"]
        self.loc_emb_size = res["loc_emb_size"]
        self.uid_emb_size = res["uid_emb_size"]
        self.voc_emb_size = res["voc_emb_size"]
        self.pretrain = 1


if __name__ == '__main__':

    np.random.seed(1)
    # torch.manual_seed(1)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument('--loc_emb_size', type=int,
                        default=50, help="location embeddings size")
    parser.add_argument('--uid_emb_size', type=int,
                        default=20, help="user id embeddings size")
    parser.add_argument('--voc_emb_size', type=int,
                        default=25, help="words embeddings size")
    parser.add_argument('--tim_emb_size', type=int,
                        default=5, help="time embeddings size")
    parser.add_argument('--hidden_size', type=int, default=50)
    parser.add_argument('--dropout_p', type=float, default=0.3)
    parser.add_argument('--data_name', type=str, default='foursquare')
    parser.add_argument('--learning_rate', type=float, default=5 * 1e-2)
    parser.add_argument('--lr_step', type=int, default=2)
    parser.add_argument('--lr_decay', type=float, default=0.5)
    parser.add_argument('--optim', type=str, default='Adam',
                        choices=['Adam', 'SGD'])
    parser.add_argument('--L2', type=float, default=1 *
                        1e-5, help=" weight decay (L2 penalty)")
    parser.add_argument('--clip', type=float, default=5.0)
    parser.add_argument('--epoch_max', type=int, default=100)
    parser.add_argument('--history_mode', type=str,
                        default='avg', choices=['max', 'avg', 'whole'])
    parser.add_argument('--rnn_type', type=str,
                        default='RNN', choices=['LSTM', 'GRU', 'RNN'])
    parser.add_argument('--attn_type', type=str, default='dot',
                        choices=['general', 'concat', 'dot'])
    parser.add_argument('--data_path', type=str, default='../data/')
    parser.add_argument('--save_path', type=str, default='../myresults/')
    parser.add_argument('--model_mode', type=str, default='simple',
                        choices=['simple', 'simple_long', 'attn_avg_long_user', 'attn_local_long'])
    parser.add_argument('--pretrain', type=int, default=0)

    args = parser.parse_args()

    run(args)

    # if args.pretrain == 1:
    #     args = load_pretrained_model(args)

    # parameters = RnnParameterData()
    # candidate = parameters.data_neural.keys()
    # data_test, test_idx = generate_input_history(parameters.data_neural, 'test', mode2=parameters.history_mode,
    #                                              candidate=candidate)

    # loc_t, target_t = loc[:], target[:]

    # print(loc_t)
    # model1 = TrajPreSimple1(hidden_dim=parameters.hidden_size,
    #                         word_dim=parameters.loc_size)

    # model2 = TrajPreSimple2(word_dim=parameters.loc_emb_size, hidden_dim=parameters.hidden_size,
    #                         total_vocabulary_size=parameters.loc_size)
    # train_with_sgd(
    #     model2, loc_t, target_t, nepoch=200, evaluate_loss_after=1)
