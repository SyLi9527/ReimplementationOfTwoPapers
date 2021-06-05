# coding: utf-8

import numpy as np
import pickle
from collections import deque, Counter
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop
import matplotlib.pyplot as plt
# import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
class RnnParameterData(object):
    def __init__(self, loc_emb_size=50, uid_emb_size=30, voc_emb_size=20, tim_emb_size=4, hidden_size=45,
                 lr=5e-3, lr_step=2, min_lr = 1e-5, lr_decay=0.1, dropout_p=0.3, L2=1e-5, clip=0.3, optim='Adam',
                 history_mode='max', attn_type='dot', epoch_max=30, rnn_type='RNN', model_mode='simple_long',
                 data_path='../data/', save_path='../results/', data_name='foursquare', plot_user_traj=-1, use_geolife_data = False):
        self.data_path = data_path
        self.save_path = save_path
        self.data_name = data_name
        data = pickle.load(
            open(self.data_path + self.data_name + '.pk', 'rb'), encoding='latin1')
        self.vid_list = data['vid_list']
        self.uid_list = data['uid_list']
        self.data_neural = data['data_neural']
        self.plot_user_traj = plot_user_traj
        self.use_geolife_data = use_geolife_data
        self.tim_size = 48
        self.loc_size = len(self.vid_list)
        self.uid_size = len(self.uid_list)
        self.loc_emb_size = loc_emb_size
        self.tim_emb_size = tim_emb_size
        self.voc_emb_size = voc_emb_size
        self.uid_emb_size = uid_emb_size
        self.hidden_size = hidden_size

        self.epoch = epoch_max
        self.dropout_p = dropout_p
        self.use_cuda = False
        self.lr = lr
        self.min_lr = min_lr
        self.lr_step = lr_step
        self.lr_decay = lr_decay
        self.optim = optim
        self.L2 = L2
        self.clip = clip

        self.attn_type = attn_type
        self.rnn_type = rnn_type
        self.history_mode = history_mode
        self.model_mode = model_mode
class Models(Layer):
    """baseline rnn model"""

    def __init__(self, parameters):
        super(Models, self).__init__()
        self.loc_size = parameters.loc_size
        self.loc_emb_size = parameters.loc_emb_size
        self.tim_size = parameters.tim_size
        self.tim_emb_size = parameters.tim_emb_size
        self.hidden_size = parameters.hidden_size
        self.uid_size = parameters.uid_size
        self.uid_emb_size = parameters.uid_emb_size
        self.input_size = self.loc_emb_size + self.tim_emb_size
        
        self.use_cuda = parameters.use_cuda
        self.attn_type = parameters.attn_type
        self.rnn_type = parameters.rnn_type
        self.model_mode = parameters.model_mode
        self.dropout_p = parameters.dropout_p


        self.emb_loc = Embedding(self.loc_size, self.loc_emb_size)
        self.emb_tim = Embedding(self.tim_size, self.tim_emb_size)
        self.emb_uid = Embedding(self.uid_size, self.uid_emb_size)
        self.emb_loc_history = Embedding(self.loc_size, self.loc_emb_size)
        self.emb_tim_history = Embedding(self.tim_size, self.tim_emb_size)
        self.dropout_input = Dropout(self.dropout_p, )
        self.dropout_output = Dropout(self.dropout_p, )
        self.activation_layer = Dense(self.hidden_size, activation='relu')
        self.fully_connnected_layer = Dense(self.loc_size, activation='softmax')
        self.fc_attn = Dense(self.hidden_size, activation='tanh')
        if self.model_mode == 'simple':
            pass
        elif self.model_mode == 'simple_long':
            # need to extend
            pass
        else:
            #need to exetnd
            pass

        if self.rnn_type == 'GRU':
            self.rnn = GRU(self.hidden_size)
        elif self.rnn_type == 'LSTM':
            self.rnn = LSTM(self.hidden_size)
        elif self.rnn_type == 'RNN':
            self.rnn = SimpleRNN(self.hidden_size)
        # self.loc_embbeding_layer = layers.embeddings(
        #     self.loc_size, self.loc_emb_size)
        # self.tim_embbeding_layer = layers.embeddings(
        #     self.tim_size, self.tim_emb_size)

def preprocess_data(data_raw, candidate):
    num = 0
    s = 0
    data = []
    positon_dict = dict()
    for user in range(42):
    # for session in data_raw:
        session = data_raw[user]
        data_row = []
        for position in session:

            if position not in positon_dict:
            
                positon_dict[position] = num
                num += 1
            else:
                pass
            data_row.append(positon_dict[position])
        data.append(data_row)
    data = np.array(data)
    print(num)

    return data, num


def generate_geolife_data(data, mode, candidate=None):
    data_train = {}
    train_idx = {}
    
    if candidate is None:
        candidate = list(range(data.shape[0]))

    sessions = []
    for u in candidate:
        trace = {}
        session = data[u][:int(0.7 * len(data[u]))] if mode == 'train' else data[u][int(0.7 * len(data[u])):]
        # sessions.append(session)
        idx = list(range(len(session)))
        # train_id = idx[: int(0.7 * len(idx))] if mode == 'train' else idx[int(0.7 * len(idx)):]
        train_id = [0, 0] if mode == 'train' else [0]
        data_train[u] = {}
        loc_np = np.reshape(
                np.array(session[:-1]), (-1, 1))
               
        tim_np = np.reshape(   
                np.array([int(i * 2)%48 for i in range(len(session[:-1]))]), (-1, 1))
        target = np.reshape(
                np.array(session[1:]), (-1, 1))
        trace['loc'] = loc_np
        trace['target'] = target
        trace['tim'] =  tim_np
        data_train[u][0] = trace
        train_idx[u] = train_id
    # print(train_idx[])
    return data_train, train_idx

def generate_input_history(data_neural, mode, mode2=None, candidate=None):
    data_train = {}
    train_idx = {}
    if candidate is None:
        candidate = data_neural.keys()
    for u in candidate:
        sessions = data_neural[u]['sessions']
        train_id = data_neural[u][mode]
        data_train[u] = {}
        for c, i in enumerate(train_id):
            if mode == 'train' and c == 0:
                continue
            session = sessions[i]
            trace = {}
            loc_np = np.reshape(
                np.array([s[0] for s in session[:-1]]), (len(session[:-1]), 1))
            # print(loc_np.shape)
            tim_np = np.reshape(
                np.array([s[1] for s in session[:-1]]), (len(session[:-1]), 1))
            # voc_np = np.reshape(np.array([s[2] for s in session[:-1]]), (len(session[:-1]), 27))
            target = np.array([s[0] for s in session[1:]])
            trace['loc'] = loc_np
            trace['target'] = target
            trace['tim'] =  tim_np
            # trace['voc'] = Variable(torch.LongTensor(voc_np))

            history = []
            if mode == 'test':
                test_id = data_neural[u]['train']
                for tt in test_id:
                    history.extend([(s[0], s[1]) for s in sessions[tt]])
            for j in range(c):
                history.extend([(s[0], s[1]) for s in sessions[train_id[j]]])
            history = sorted(history, key=lambda x: x[1], reverse=False)

            # merge traces with same time stamp
            if mode2 == 'max':
                history_tmp = {}
                for tr in history:
                    if tr[1] not in history_tmp:
                        history_tmp[tr[1]] = [tr[0]]
                    else:
                        history_tmp[tr[1]].append(tr[0])
                history_filter = []
                for t in history_tmp:
                    if len(history_tmp[t]) == 1:
                        history_filter.append((history_tmp[t][0], t))
                    else:
                        tmp = Counter(history_tmp[t]).most_common()
                        if tmp[0][1] > 1:
                            history_filter.append((history_tmp[t][0], t))
                        else:
                            ti = np.random.randint(len(tmp))
                            history_filter.append((tmp[ti][0], t))
                history = history_filter
                history = sorted(history, key=lambda x: x[1], reverse=False)
            elif mode2 == 'avg':
                history_tim = [t[1] for t in history]
                history_count = [1]
                last_t = history_tim[0]
                count = 1
                for t in history_tim[1:]:
                    if t == last_t:
                        count += 1
                    else:
                        history_count[-1] = count
                        history_count.append(1)
                        last_t = t
                        count = 1
            ################

            history_loc = np.reshape(
                np.array([s[0] for s in history]), (len(history), 1))
            history_tim = np.reshape(
                np.array([s[1] for s in history]), (len(history), 1))
            trace['history_loc'] = history_loc
            trace['history_tim'] = history_tim
            if mode2 == 'avg':
                trace['history_count'] = history_count

            data_train[u][i] = trace
        train_idx[u] = train_id
    return data_train, train_idx


def generate_input_long_history2(data_neural, mode, candidate=None):
    data_train = {}
    train_idx = {}
    if candidate is None:
        candidate = data_neural.keys()
    for u in candidate:
        sessions = data_neural[u]['sessions']
        train_id = data_neural[u][mode]
        data_train[u] = {}

        trace = {}
        session = []
        for c, i in enumerate(train_id):
            session.extend(sessions[i])
        target = np.array([s[0] for s in session[1:]])

        loc_tim = []
        loc_tim.extend([(s[0], s[1]) for s in session[:-1]])
        loc_np = np.reshape(np.array([s[0]
                                      for s in loc_tim]), (len(loc_tim), 1))
        tim_np = np.reshape(np.array([s[1]
                                      for s in loc_tim]), (len(loc_tim), 1))
        # print(loc_np.shape)
        trace['loc'] = loc_np
        trace['tim'] = tim_np
        trace['target'] = target
        data_train[u][i] = trace
        # train_idx[u] = train_id
        if mode == 'train':
            train_idx[u] = [0, i]
        else:
            train_idx[u] = [i]
    return data_train, train_idx


def generate_input_long_history(data_neural, mode, candidate=None):
    data_train = {}
    train_idx = {}
    if candidate is None:
        candidate = data_neural.keys()
    for u in candidate:
        sessions = data_neural[u]['sessions']
        train_id = data_neural[u][mode]
        data_train[u] = {}
        for c, i in enumerate(train_id):
            trace = {}
            if mode == 'train' and c == 0:
                continue
            session = sessions[i]
            target = np.array([s[0] for s in session[1:]])

            history = []
            if mode == 'test':
                test_id = data_neural[u]['train']
                for tt in test_id:
                    history.extend([(s[0], s[1]) for s in sessions[tt]])
            for j in range(c):
                history.extend([(s[0], s[1]) for s in sessions[train_id[j]]])

            history_tim = [t[1] for t in history]
            history_count = [1]
            last_t = history_tim[0]
            count = 1
            for t in history_tim[1:]:
                if t == last_t:
                    count += 1
                else:
                    history_count[-1] = count
                    history_count.append(1)
                    last_t = t
                    count = 1

            history_loc = np.reshape(
                np.array([s[0] for s in history]), (len(history), 1))
            # print(history_loc.shape)
            history_tim = np.reshape(
                np.array([s[1] for s in history]), (len(history), 1))
            trace['history_loc'] = history_loc
            trace['history_tim'] =  history_tim
            trace['history_count'] = history_count

            loc_tim = history
            loc_tim.extend([(s[0], s[1]) for s in session[:-1]])
            loc_np = np.reshape(
                np.array([s[0] for s in loc_tim]), (len(loc_tim), 1))
            tim_np = np.reshape(
                np.array([s[1] for s in loc_tim]), (len(loc_tim), 1))
            trace['loc'] = loc_np
            trace['tim'] =  tim_np
            trace['target'] =target
            data_train[u][i] = trace
        train_idx[u] = train_id
    return data_train, train_idx

def generate_queue(train_idx, mode, mode2):
    """return a deque. You must use it by train_queue.popleft()"""
    user = list(train_idx.keys())
    train_queue = deque()
    if mode == 'random':
        initial_queue = {}
        for u in user:
            if mode2 == 'train':
                initial_queue[u] = deque(train_idx[u][1:])
            else:
                initial_queue[u] = deque(train_idx[u])
        queue_left = 1
        while queue_left > 0:
            np.random.shuffle(user)
            for j, u in enumerate(user):
                if len(initial_queue[u]) > 0:
                    train_queue.append((u, initial_queue[u].popleft()))
                if j >= int(0.01 * len(user)):
                    break
            queue_left = sum([1 for x in initial_queue if len(initial_queue[x]) > 0])
    elif mode == 'normal':
        for u in user:
            for i in train_idx[u]:
                train_queue.append((u, i))
    return train_queue

def get_acc(target, scores):
    """target and scores are torch cuda Variable"""
    target = target.data.cpu().numpy()
    val, idxx = scores.data.topk(10, 1)
    predx = idxx.cpu().numpy()
    acc = np.zeros((3, 1))
    for i, p in enumerate(predx):
        t = target[i]
        if t in p[:10] and t > 0:
            acc[0] += 1
        if t in p[:5] and t > 0:
            acc[1] += 1
        if t == p[0] and t > 0:
            acc[2] += 1
    return acc


def get_hint(target, scores, users_visited):
    """target and scores are torch cuda Variable"""
    target = target.data.cpu().numpy()
    val, idxx = scores.data.topk(1, 1)
    predx = idxx.cpu().numpy()
    hint = np.zeros((3,))
    count = np.zeros((3,))
    count[0] = len(target)
    for i, p in enumerate(predx):
        t = target[i]
        if t == p[0] and t > 0:
            hint[0] += 1
        if t in users_visited:
            count[1] += 1
            if t == p[0] and t > 0:
                hint[1] += 1
        else:
            count[2] += 1
            if t == p[0] and t > 0:
                hint[2] += 1
    return hint, count

# optimizer, criterion, clip, lr
def run_simple(data, run_idx, mode, mode2=None):
    """mode=train: return model, avg_loss
       mode=test: return avg_loss,avg_acc,users_rnn_acc"""
    run_queue = None
    if mode == 'train':
        run_queue = generate_queue(run_idx, 'random', 'train')
    elif mode == 'test':
        run_queue = generate_queue(run_idx, 'normal', 'test')
    total_loss = []
    queue_len = len(run_queue)

    users_acc = {}
    loc, target, tim, history_loc, history_tim, history_count = \
        np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    uid = list()
    for c in range(queue_len):
        u, i = run_queue.popleft()
        if u not in users_acc:
            users_acc[u] = [0, 0]
        loc = np.append(loc, data[u][i]['loc'])
        # print(loc.shape)
        tim = np.append(tim, data[u][i]['tim'])
        target = np.append(target, data[u][i]['target'])
        uid += [u]

        if 'attn' in mode2:
            history_loc = np.append(history_loc, data[u][i]['history_loc'])
            history_tim = np.append(history_tim, data[u][i]['history_tim'])


        if mode2 == 'attn_avg_long_user':
            history_count = np.append(history_count, data[u][i]['history_count'])
            target_len = target.data.size()[0]

            # scores = model(loc, tim, history_loc, history_tim,
            #                history_count, uid, target_len)
        elif mode2 == 'attn_local_long':
            target_len = target.data.size()[0]
            # scores = model(loc, tim, target_len)
    if 'simple' in  mode2:
        return loc, tim, target
    elif mode2 == 'attn_local_long':
        pass
    elif mode2 == 'attn_avg_long_user':
        pass
def run_simple_mod(data, run_idx, mode, use_geolife_data, mode2=None):
    """mode=train: return model, avg_loss
       mode=test: return avg_loss,avg_acc,users_rnn_acc"""
    run_queue = None
    if mode == 'train':
        run_queue = generate_queue(run_idx, 'random', 'train')
    elif mode == 'test':
        run_queue = generate_queue(run_idx, 'normal', 'test')
    total_loss = []
    queue_len = len(run_queue)

    users_acc = {}
    users = []
    loc, target, tim, history_loc, history_tim, history_count = \
        [], [], [], [], [], []
    uid = list()
    target_len = []
    for c in range(queue_len):
        u, i = run_queue.popleft()
        # if use_geolife_data:
        #     i = 0
        if u not in users_acc:
            users_acc[u] = [0, 0]
        users.append(u)
        loc.append(data[u][i]['loc'])
        # print(loc.shape)
        tim.append(data[u][i]['tim'])
        target.append(data[u][i]['target'])
        uid.append(np.array([u]))

        if 'attn' in mode2:
            history_loc.append(data[u][i]['history_loc'])
            history_tim.append(data[u][i]['history_tim'])


        if mode2 == 'attn_avg_long_user':
            history_count.append(np.array(data[u][i]['history_count']))
            target_len.append(np.array([data[u][i]['target'].shape[0]]))

            # scores = model(loc, tim, history_loc, history_tim,
            #                history_count, uid, target_len)
        elif mode2 == 'attn_local_long':
            target_len.append(np.array([data[u][i]['target'].shape[0]]))
#             # scores = model(loc, tim, target_len)
    if 'simple' in  mode2:
        return  users_acc, users, loc, tim, target
    elif mode2 == 'attn_local_long':
        return  users_acc, users, loc, tim, target_len, target
    elif mode2 == 'attn_avg_long_user':
        return  users_acc, users, loc, tim, uid, history_loc, history_tim, history_count, target_len, target


def markov(parameters, candidate):
    validation = {}
    for u in candidate:
        traces = parameters.data_neural[u]['sessions']
        train_id = parameters.data_neural[u]['train']
        test_id = parameters.data_neural[u]['test']
        trace_train = []
        for tr in train_id:
            trace_train.append([t[0] for t in traces[tr]])
        locations_train = []
        for t in trace_train:
            locations_train.extend(t)
        trace_test = []
        for tr in test_id:
            trace_test.append([t[0] for t in traces[tr]])
        locations_test = []
        for t in trace_test:
            locations_test.extend(t)
        validation[u] = [locations_train, locations_test]
    acc = 0
    count = 0
    user_acc = {}
    for u in validation.keys():
        topk = list(set(validation[u][0]))
        transfer = np.zeros((len(topk), len(topk)))

        # train
        sessions = parameters.data_neural[u]['sessions']
        train_id = parameters.data_neural[u]['train']
        for i in train_id:
            for j, s in enumerate(sessions[i][:-1]):
                loc = s[0]
                target = sessions[i][j + 1][0]
                if loc in topk and target in topk:
                    r = topk.index(loc)
                    c = topk.index(target)
                    transfer[r, c] += 1
        for i in range(len(topk)):
            tmp_sum = np.sum(transfer[i, :])
            if tmp_sum > 0:
                transfer[i, :] = transfer[i, :] / tmp_sum

        # validation
        user_count = 0
        user_acc[u] = 0
        test_id = parameters.data_neural[u]['test']
        for i in test_id:
            for j, s in enumerate(sessions[i][:-1]):
                loc = s[0]
                target = sessions[i][j + 1][0]
                count += 1
                user_count += 1
                if loc in topk:
                    pred = np.argmax(transfer[topk.index(loc), :])
                    if pred >= len(topk) - 1:
                        pred = np.random.randint(len(topk))

                    pred2 = topk[pred]
                    if pred2 == target:
                        acc += 1
                        user_acc[u] += 1
        user_acc[u] = user_acc[u] / user_count
    avg_acc = np.mean([user_acc[u] for u in user_acc])
    return avg_acc, user_acc

def generator_simple(inputs):
    loc, tim, target = inputs
    loc_len = len(loc)
    for i in range(loc_len):
        loc_each = loc[i]
        tim_each = tim[i]
        target_each = target[i]
        yield [loc_each, tim_each], target_each
def generator_simple_inputs(inputs):
    loc, tim = inputs
    loc_len = len(loc)
    for i in range(loc_len):
        loc_each = loc[i]
        tim_each = tim[i]

        yield [loc_each, tim_each]
def generator_target(target):
    for i in target:
        yield i.reshape(-1, 1)
def generator_attn_user(inputs):
    loc, tim, uid, hloc, htim, hcount, target_len, target = inputs
    loc_len = len(loc)
    for i in range(loc_len):
        loc_each = loc[i]
        tim_each = tim[i]
        target_len_each = target_len[i].reshape(-1, 1)
        target_each = target[i]

        uid_each = uid[i].reshape(-1, 1)
        hloc_each = hloc[i]
        htim_each = htim[i]
        hcount_each = hcount[i].reshape(-1, 1)
        yield [loc_each, tim_each, uid_each, hloc_each, htim_each, hcount_each, target_len_each], target_each
def generator_attn_user_inputs(inputs):
    loc, tim, uid, hloc, htim, hcount, target_len = inputs
    loc_len = len(loc)
    for i in range(loc_len):
        loc_each = loc[i].reshape(1, -1)
        tim_each = tim[i].reshape(1, -1)
        target_len_each = target_len[i].reshape(1, -1)

        uid_each = uid[i].reshape(1, -1)
        hloc_each = hloc[i].reshape(1, -1)
        htim_each = htim[i].reshape(1, -1)
        hcount_each = hcount[i].reshape(1, -1)
        yield [loc_each, tim_each, uid_each, hloc_each, htim_each, hcount_each, target_len_each]
def generator_attn_long(inputs):
    loc, tim, target_len, target = inputs
    loc_len = len(loc)
    for i in range(loc_len):
        loc_each = loc[i]
        tim_each = tim[i]
        target_len_each = target_len[i].reshape(-1, 1)
        target_each = target[i]
        yield [loc_each, tim_each, target_len_each], target_each
def generator_attn_long_inputs(inputs):
    loc, tim, target_len = inputs
    loc_len = len(loc)
    for i in range(loc_len):
        loc_each = loc[i].reshape(1, -1)
        tim_each = tim[i].reshape(1, -1)
        target_len_each = target_len[i].reshape(1, -1)

        yield [loc_each, tim_each, target_len_each]
def generator_test(inputs):
    return [inputs[:-1]], inputs[-1]
# def whole_accuracy(target, pred):
#     return np.sum((target - pred.numpy()) == 0) / target.shape[0]

def train_model(model, data, idx, model_mode, reduce_lr, user, use_geolife_data, Train):
    # We keep track of the losses so we can plot them later
    if Train:
        args = run_simple_mod(data, idx, 'train', use_geolife_data, model_mode)
    else:
        if user == -1:
            args = run_simple_mod(data, idx, 'test', use_geolife_data, model_mode)
        else:
            args = run_simple_mod(data, idx, 'train', use_geolife_data, model_mode)
    # if model_mode == 'attn_avg_long_user':
    #     data_generator = generator_attn_user(args[2:])
    # elif model_mode == 'attn_local_long':
    #     data_generator = generator_attn_long(args[2:])
    # else:
    #     data_generator = generator_simple(args[2:])

    if Train:
        # steps_per_epoch= 200
        if model_mode == 'attn_avg_long_user':
            data_generator = generator_attn_user(args[2:])
        elif model_mode == 'attn_local_long':
            data_generator = generator_attn_long(args[2:])
        else:
            data_generator = generator_simple(args[2:])
        

        history = model.fit(data_generator, epochs=1, callbacks=[reduce_lr])
        return model, history.history
    else:
        scce = tf.keras.losses.SparseCategoricalCrossentropy()
        if model_mode == 'attn_avg_long_user':
            data_generator = generator_attn_user_inputs(args[2: -1])
        elif model_mode == 'attn_local_long':
            data_generator = generator_attn_long_inputs(args[2: -1])
        else:
            data_generator = generator_simple_inputs(args[2: -1])
        data_target = generator_target(args[-1])
        users_acc, users = args[0], args[1]
        users_rnn_acc = {}
        loss = []
        lens_targets = [len(x) for x in args[-1]]
        # max_index = lens_targets.index(max(lens_targets))
        len_target = len(args[-1])
        match = total = 0

        if -1 < user < 886 and not use_geolife_data:
        # plot trajectory
            out = model.predict(data_generator)
            out = tf.nn.softmax(out, axis = 1)
            tar = []
            for i in range(len_target):
                tar.extend(args[-1][i].tolist())
            y_pred = np.argmax(out, axis = 1)
            plt.figure("predict traj and real traj")
            ax = plt.gca()
            ax.set_xlabel('time steps')
            ax.set_ylabel('position')
            ax.set_yscale('log')
            ax.scatter(np.arange(np.sum(lens_targets)), tar, c='blue', s=5, alpha=0.5, label="$real$")
            ax.scatter(np.arange(np.sum(lens_targets)), y_pred, c='red', s=5, alpha=0.5, label="$predict$")
            plt.legend()
            plt.savefig(model_mode)
            # ax.scatter(x_list, y_list, c='r', s=20, alpha=0.5)
            # plt.show()
        else:
            for i in range(len_target):
        
                # result = model.test_on_batch(next(data_generator), next(data_target))
                out = model.predict_on_batch([item[i].reshape(1, -1) for item in args[2: -1]])
                out = tf.nn.softmax(out, axis = 1)
                # if i == max_index:
                #     y_pred = np.argmax(out, axis = 1)
                #     plt.figure("predict traj and real traj")
                #     ax = plt.gca()
                #     ax.set_xlabel('time steps')
                #     ax.set_ylabel('position')
                #     ax.set_yscale('log')
                #     ax.scatter(np.arange(), y_list, c='blue', s=20, alpha=0.5)
                #     plt.savefig("attn_local_long")
                per_loss = scce(args[-1][i], out).numpy()
                # print(args[-1][i] - np.argmax(out, axis = 1) == 0)
                # match = np.sum(args[-1][i] == np.argmax(out, axis = 1))
                match = np.sum(args[-1][i] == np.reshape(np.argmax(out, axis = 1), (-1, 1)))
                # total += args[-1][i].shape[0]
                users_acc[users[i]][0] += lens_targets[i]
                users_acc[users[i]][1] += match
                loss.append(per_loss)

            for u in users_acc:
                tmp_acc = users_acc[u][1] / users_acc[u][0]
                users_rnn_acc[u] = tmp_acc
            avg_acc = np.mean([users_rnn_acc[x] for x in users_rnn_acc], dtype=np.float32)
            # avg_acc = np.float32(match / total)
            avg_loss = np.mean(loss, dtype=np.float32)
            # result = model.evaluate(data_generator)
            # avg_loss = result[1]
            # avg_acc = result[2]

            return avg_loss, avg_acc