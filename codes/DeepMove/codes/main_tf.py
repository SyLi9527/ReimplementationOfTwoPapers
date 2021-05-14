import numpy as np
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import time
import json
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau

from train_tf import RnnParameterData
from json import encoder

from model_tf import TrajPreSimple, TrajPreAttnAvgLongUser, TrajPreLocalAttnLong
from train_tf import train_model, RnnParameterData, generate_input_history, markov, \
    generate_input_long_history, generate_input_long_history2
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
        model = TrajPreSimple(parameters=parameters)
    elif parameters.model_mode == 'attn_avg_long_user':
        model = TrajPreAttnAvgLongUser(parameters=parameters)
    elif parameters.model_mode == 'attn_local_long':
        model = TrajPreLocalAttnLong(parameters=parameters)
    if args.pretrain == 1:
        pass
    model.compile(
            optimizer = Adam(
                learning_rate=parameters.lr,
                clipvalue=parameters.clip
            ),
            run_eagerly=True,
            loss = tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[
                tf.keras.losses.SparseCategoricalCrossentropy(),
                tf.keras.metrics.SparseCategoricalAccuracy(),
    ])
    reduce_lr = ReduceLROnPlateau(monitor= 'sparse_categorical_crossentropy', factor=parameters.lr_decay,
                              patience=parameters.lr_step, min_lr=parameters.min_lr)
    candidate = parameters.data_neural.keys()
    avg_acc_markov, users_acc_markov = markov(parameters, candidate)
    metrics = {'train_loss': [], 'valid_loss': [],
               'accuracy': [], 'valid_acc': {}, 'lr': []}
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
    # SAVE_PATH = args.save_path
    # tmp_path = 'checkpoint/'
    # os.makedirs(SAVE_PATH + tmp_path, exist_ok=True)
    show_per_epoch = 2
    lr_last = lr = parameters.lr
    checkpoint_path = "training/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    for epoch in range(parameters.epoch):

        if args.pretrain == 0:
            if lr > lr_last:
                model.load_weights(checkpoint_path.format(epoch=np.argmax(metrics['accuracy'])))
                # model, history = train_model(model, data_train, train_idx, parameters.model_mode, reduce_lr, Train=True, Back=None)
            else:
                cp_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path.format(epoch=epoch), 
                verbose=1, 
                save_weights_only=True)
                model, history = train_model(model, data_train, train_idx, parameters.model_mode, reduce_lr, Train=True, Back=cp_callback)
            # Include the epoch in the file name (uses `str.format`)

            # Create a callback that saves the model's weights every 5 epochs
            
            # Create a new model instance  ???
            # Save the weights using the `checkpoint_path` format
            # model.save_weights(checkpoint_path.format(epoch=epoch))

            # loss', 'sparse_categorical_crossentropy', 'sparse_categorical_accuracy', 'lr
            lr_last, lr = lr, history['lr'][0]
            if not (epoch % show_per_epoch):
                result = train_model(model, data_test, test_idx, parameters.model_mode, reduce_lr, Train=False, Back=None)
        metrics['lr'].append(lr)
        metrics['train_loss'].extend(history['sparse_categorical_crossentropy'])
        metrics['valid_loss'].append(result[1])
        metrics['accuracy'].append(result[2])
        # metrics['valid_acc'][epoch] = users_acc
    model.save('saved_model/my_model')
    save_name = 'res'
    json.dump({'args': argv, 'metrics': metrics}, fp=open(
        args.save_path + save_name + '.rs', 'w'), indent=4)
    metrics_view = {'train_loss': [], 'valid_loss': [], 'accuracy': []}
    for key in metrics_view:
        metrics_view[key] = metrics[key]
    json.dump({'args': argv, 'metrics': metrics_view}, fp=open(
        args.save_path + save_name + '.txt', 'w'), indent=4)
    
def load_pretrained_model(args):
    pass

if __name__ == '__main__':
    np.random.seed(1)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument('--loc_emb_size', type=int,
                        default=40, help="location embeddings size")
    parser.add_argument('--uid_emb_size', type=int,
                        default=20, help="user id embeddings size")
    parser.add_argument('--voc_emb_size', type=int,
                        default=25, help="words embeddings size")
    parser.add_argument('--tim_emb_size', type=int,
                        default=5, help="time embeddings size")
    parser.add_argument('--hidden_size', type=int, default=50)
    parser.add_argument('--dropout_p', type=float, default=0.3)
    parser.add_argument('--data_name', type=str, default='foursquare')
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--lr_step', type=int, default=2)
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--optim', type=str, default='Adam',
                        choices=['Adam', 'SGD'])
    parser.add_argument('--L2', type=float, default=1 *
                        1e-5, help=" weight decay (L2 penalty)")
    parser.add_argument('--clip', type=float, default=0.3)
    parser.add_argument('--epoch_max', type=int, default=20)
    parser.add_argument('--history_mode', type=str,
                        default='avg', choices=['max', 'avg', 'whole'])
    parser.add_argument('--rnn_type', type=str,
                        default='LSTM', choices=['LSTM', 'GRU', 'RNN'])
    parser.add_argument('--attn_type', type=str, default='concat',
                        choices=['general', 'concat', 'dot'])
    parser.add_argument('--data_path', type=str, default='../data/')
    parser.add_argument('--save_path', type=str, default='../results/')
    parser.add_argument('--model_mode', type=str, default='attn_local_long',
                        choices=['simple', 'simple_long', 'attn_avg_long_user', 'attn_local_long'])
    parser.add_argument('--pretrain', type=int, default=0)
    parser.add_argument('--min-lr', type=float, default=1e-5)
    args = parser.parse_args()
    if args.pretrain == 1:
        args = load_pretrained_model(args)

    ours_acc = run(args)