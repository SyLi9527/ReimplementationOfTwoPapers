from numpy.core.fromnumeric import repeat
from numpy.core.numeric import outer
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau
import os
import numpy as np
from train_tf import generator_simple, train_model, RnnParameterData, generate_input_history, markov, \
    generate_input_long_history, generate_input_long_history2, generator_attn_user

# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()

from train_tf import run_simple, generate_input_history, run_simple_mod, \
    generate_input_long_history, generate_input_long_history2
from train_tf import RnnParameterData
cpu = tf.config.experimental.list_physical_devices("CPU")
# tf.config.experimental.set_memory_growth(cpu[0], True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



class RNNcell(Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(RNNcell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = K.dot(inputs, self.kernel)
        output = h + K.dot(prev_output, self.recurrent_kernel)
        return output, [output]

class MyAttn(Layer):
    def __init__(self, method, hidden_size):
        super(MyAttn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = Dense(self.hidden_size)
        elif self.method == 'concat':
            self.attn = Dense(self.hidden_size * 2)
            self.other = tf.Variable(self.hidden_size, trainable = False)

    def call(self, out_state, history):
        out_state = out_state.numpy()
        history = history.numpy()
        seq_len = history.shape[0]
        state_len = out_state.shape[0]
        attn_energies = np.zeros((state_len, seq_len))
        # attn_energies = tf.Variable(tf.zeros(state_len, seq_len, tf.float32))
        for i in range(state_len):
            for j in range(seq_len):
                if self.method == 'dot':
                    energy = out_state[i].dot(history[j])
                    attn_energies[i, j] = energy
                else:
                    pass
        attn_energies = tf.dtypes.cast(attn_energies, tf.float32)
        score = tf.nn.softmax(attn_energies, axis = 1)
        return score

class History(Layer):
    
    def call(self, loc_history, tim_history, history_count):
        loc_history = loc_history.numpy()
        tim_history = tim_history.numpy()
        history_count_len = history_count.shape[0]
        history_count = history_count.numpy().tolist()
        # history_count_len = len(history_count)
        loc_history2 = np.zeros((history_count_len, loc_history.shape[-1]))
        tim_history2 = np.zeros((history_count_len, tim_history.shape[-1]))
        count = 0
        for i, c in enumerate(history_count):
            c = c[0]
            if c == 1:
                tmp = loc_history[count].reshape(1, -1)
            else:
                tmp = np.mean(loc_history[count:count + c, :], axis=0).reshape(1, -1)
            loc_history2[i, :] = tmp
            tim_history2[i, :] = tim_history[count, :].reshape(1, -1)
            count += c

        loc_history2 = tf.dtypes.cast(loc_history2,tf.float32)
        tim_history2 = tf.dtypes.cast(tim_history2,tf.float32)
        history = concatenate((loc_history2, tim_history2), axis = 1)
        return history
        
class TrajPreSimple(Model):
    def __init__(self,  parameters):
        super(TrajPreSimple, self).__init__()
        self.loc_size = parameters.loc_size
        self.loc_emb_size = parameters.loc_emb_size
        self.tim_size = parameters.tim_size
        self.tim_emb_size = parameters.tim_emb_size
        self.hidden_size = parameters.hidden_size
        self.uid_size = parameters.uid_size
        self.uid_emb_size = parameters.uid_emb_size
        self.input_size = self.loc_emb_size + self.tim_emb_size
        
        self.attn_type = parameters.attn_type
        self.rnn_type = parameters.rnn_type
        self.model_mode = parameters.model_mode
        self.dropout_p = parameters.dropout_p


        self.emb_loc = Embedding(self.loc_size, self.loc_emb_size)
        self.emb_tim = Embedding(self.tim_size, self.tim_emb_size)

        self.dropout_input = Dropout(self.dropout_p, )
        self.dropout_output = Dropout(self.dropout_p, )
        self.activation_layer = Dense(self.hidden_size, activation='relu', kernel_initializer='random_normal',
    bias_initializer='zeros')
        self.fully_connnected_layer = Dense(self.loc_size, activation='softmax', activity_regularizer=tf.keras.regularizers.l2(0.0001))
        self.fc_attn = Dense(self.hidden_size, activation='tanh', activity_regularizer=tf.keras.regularizers.l2(0.0001), kernel_initializer='random_uniform',
    bias_initializer='zeros')
        
        if self.rnn_type == 'GRU':
            self.rnn = GRU(self.hidden_size)
        elif self.rnn_type == 'LSTM':
            self.rnn = LSTM(self.hidden_size)
        elif self.rnn_type == 'RNN':
            self.rnn = SimpleRNN(self.hidden_size)
        # self.init_weights()
    # def init_weights(self):
    #     """
    #     Here we reproduce Keras default initialization weights for consistency with Keras version
    #     """
    #     ih = (param.data for name, param in self.named_parameters()
    #           if 'weight_ih' in name)
    #     hh = (param.data for name, param in self.named_parameters()
    #           if 'weight_hh' in name)
    #     b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        # for t in ih:
        #     tf.nn.init.xavier_uniform_(t)
        # for t in hh:
        #     tf.nn.init.orthogonal_(t)
        # for t in b:
        #     tf.nn.init.constant_(t, 0)

    def call(self, inputs):
        loc_input, tim_input = inputs
        loc_input = tf.reshape(loc_input, [-1, 1])
        tim_input = tf.reshape(tim_input, [-1, 1])

        input_layer = concatenate([self.emb_loc(loc_input), self.emb_tim(tim_input)], axis=2)
        input = self.dropout_input(input_layer)

        out = self.rnn(input)
        out = self.activation_layer(out)
        out = self.dropout_output(out)
        out = self.fully_connnected_layer(out)

        return out

class TrajPreAttnAvgLongUser(Model):
    def __init__(self, parameters):
        super(TrajPreAttnAvgLongUser, self).__init__()
        self.loc_size = parameters.loc_size
        self.loc_emb_size = parameters.loc_emb_size
        self.tim_size = parameters.tim_size
        self.tim_emb_size = parameters.tim_emb_size
        self.hidden_size = parameters.hidden_size
        self.uid_size = parameters.uid_size
        self.uid_emb_size = parameters.uid_emb_size
        self.input_size = self.loc_emb_size + self.tim_emb_size
        
        self.method = parameters.attn_type
        self.use_cuda = parameters.use_cuda
        self.attn_type = parameters.attn_type
        self.attn = MyAttn(self.method, self.hidden_size)
        
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
        if self.rnn_type == 'GRU':
            self.rnn = GRU(self.hidden_size)
        elif self.rnn_type == 'LSTM':
            self.rnn = LSTM(self.hidden_size)
        elif self.rnn_type == 'RNN':
            self.rnn = SimpleRNN(self.hidden_size)

        if self.model_mode == 'simple':
            pass
        elif self.model_mode == 'simple_long':
            # need to extend
            pass
        else:
            #need to exetnd
            pass
    
    def call(self, inputs):
        loc, tim, uid, hloc, htim, hcount, target_len = inputs

        loc = tf.reshape(loc, [-1, 1])
        tim = tf.reshape(tim, [-1, 1])
        uid = tf.reshape(uid, [-1, 1])
        hloc = tf.reshape(hloc, [-1, 1])
        htim = tf.reshape(htim, [-1, 1])
        hcount = tf.reshape(hcount, [-1, 1])
        target_len = tf.reshape(target_len, [-1, 1])


        current_input = self.dropout_input(concatenate([self.emb_loc(loc), \
            self.emb_tim(tim)], axis=2))
        # add a layer here
        loc_history = tf.squeeze(self.emb_loc_history(hloc), axis = 1)
        tim_history = tf.squeeze(self.emb_tim_history(htim), axis = 1)

        # history_input = tf.squeeze(concatenate([self.emb_loc_history(hloc), \
        #     self.emb_tim_history(htim)], axis=2), axis = 1)
        loc_history = loc_history.numpy()
        tim_history = tim_history.numpy()
        history_count_len = hcount.shape[0]
        history_count = hcount.numpy().tolist()
        # history_count_len = len(history_count)
        loc_history2 = np.zeros((history_count_len, loc_history.shape[-1]))
        tim_history2 = np.zeros((history_count_len, tim_history.shape[-1]))
        count = 0
        for i, c in enumerate(history_count):
            c = c[0]
            if c == 1:
                tmp = loc_history[count].reshape(1, -1)
            else:
                tmp = np.mean(loc_history[count:count + c, :], axis=0).reshape(1, -1)
            loc_history2[i, :] = tmp
            tim_history2[i, :] = tim_history[count, :].reshape(1, -1)
            count += c

        loc_history2 = tf.dtypes.cast(loc_history2,tf.float32)
        tim_history2 = tf.dtypes.cast(tim_history2,tf.float32)
        history_input = concatenate((loc_history2, tim_history2), axis = 1)
   
        history_input = Dense(self.hidden_size, activation='tanh')(history_input)
        
        out_state = self.rnn(current_input)
        attn_weights = self.attn(out_state[-int(target_len):], history_input)
        attn_weights = tf.expand_dims(attn_weights, axis = 0)
        # history_input = tf.expand_dims(attn_weights, axis = 0)
        context = tf.squeeze(attn_weights @ history_input, axis = 0)
        out = concatenate([out_state[-int(target_len):], context], axis = 1)
        uid_emb = repeat(self.emb_uid(uid), repeats = [int(target_len)], axis = 0)
        uid_emb = tf.squeeze(uid_emb, axis = 1)
        out = concatenate([out, uid_emb], axis = 1)
        out = self.dropout_output(out)
        score = self.fully_connnected_layer(out)
        # may need to truncate score to score[-target_len, ]
        score = score[-int(target_len):, ]
        return score
        

   
        # loc_history_emb = tf.squeeze(self.emb_loc_history(loc_history_input), axis = 1)
        # tim_history_emb = tf.squeeze(self.emb_tim_history(tim_history_input), axis = 1)
        # # history_layer = MyHistoryLayer()
        # history_input = concatenate([loc_history_emb, tim_history_emb], axis=1)
        # # history_input = history_layer(loc_history_emb, tim_history_emb, history_count)
        # history_input = tf.expand_dims(self.fc_attn(history_input), axis = 2)
     
        # out_state =self.rnn(current_input)
        # # out_state = tf.reshape(out_state, (1, -1, self.hidden_size))
        # history_input = tf.reshape(history_input, ( -1, self.hidden_size))
        # # attn_weights = Attention()([out_state, history_input], return_attention_scores = True)
        # # attn_weights =
        # #  tf.expand_dims(attn_weights, axis = 0)
        # # history_input = tf.reshape(history_input, (-1, 1, self.hidden_size))
        # # out_state = tf.expand_dims(out_state, axis = 0)
        # # context = attn_weights @ history_input
        # context = DotProductAttention(out_state, history_input)
        # out = concatenate([out_state, context], axis = 1)
        # uid_emb = self.emb_uid(uid_input)
        # out = tf.reshape(out, (-1, 1, self.hidden_size))
        # out = concatenate([out, uid_emb], axis=1)
        # out = self.dropout_output(out)
                        
        # model = Model([loc_input, tim_input, uid_input, loc_history_input, tim_history_input, history_count], 
        #     outputs = out)
        # model.summary()
        # return model

class TrajPreLocalAttnLong(Model):
    def __init__(self, parameters):
        super( TrajPreLocalAttnLong, self).__init__()
        self.loc_size = parameters.loc_size
        self.loc_emb_size = parameters.loc_emb_size
        self.tim_size = parameters.tim_size
        self.tim_emb_size = parameters.tim_emb_size
        self.hidden_size = parameters.hidden_size
        self.uid_size = parameters.uid_size
        self.uid_emb_size = parameters.uid_emb_size
        self.input_size = self.loc_emb_size + self.tim_emb_size
        
        self.method = parameters.attn_type
        self.use_cuda = parameters.use_cuda
        self.attn_type = parameters.attn_type
        self.attn = MyAttn(self.method, self.hidden_size)
        self.rnn_type = parameters.rnn_type
        self.model_mode = parameters.model_mode
        self.dropout_p = parameters.dropout_p

        # need to code further
        # init_weight()
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

        if self.rnn_type == 'GRU':
            self.rnn_encoder = GRU(self.hidden_size)
            self.rnn_decoder = GRU(self.hidden_size)
        elif self.rnn_type == 'LSTM':
            self.rnn_encoder = LSTM(self.hidden_size)
            self.rnn_decoder = LSTM(self.hidden_size)
        elif self.rnn_type == 'RNN':
            self.rnn_encoder = SimpleRNN(self.hidden_size)
            self.rnn_decoder = LSTM(self.hidden_size)
    def call(self, inputs):
        loc, tim, target_len = inputs
        loc = tf.reshape(loc, [-1, 1])
        tim = tf.reshape(tim, [-1, 1])
        target_len = tf.reshape(target_len, [-1, 1])

        current_input = self.dropout_input(concatenate([self.emb_loc(loc), \
            self.emb_tim(tim)], axis=2))
        # ?? LSTM and the other difference
        hidden_history = self.rnn_encoder(current_input[:-int(target_len)])
        hidden_state = self.rnn_decoder(current_input[-int(target_len):])
        attn_weights = self.attn(hidden_state, hidden_history)
        attn_weights = tf.expand_dims(attn_weights, axis = 0)
        context = tf.squeeze(attn_weights @ hidden_history, axis = 0)

        out = concatenate([hidden_state, context], axis = 1)
        out = self.dropout_output(out)
        score = self.fully_connnected_layer(out)
        # may need to truncate score to score[-target_len, ]
        score = score[-int(target_len):, ]
        return score
    


if __name__ == '__main__':
  

    parameters = RnnParameterData()
    # loc = np.array([1, 2, 3, 5]).reshape(4, 1)
    # tim = np.array([2, 3, 4, 5]).reshape(4, 1)

    # target = np.array([0, 0, 0, 1])
    model = TrajPreSimple(parameters)
    
    

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
#     gradients = tape.gradient(loss, model.trainable_variables)
# optimizer.apply_gradients([
#     (grad, var) 
#     for (grad, var) in zip(gradients, model.trainable_variables) 
#     if grad is not None
# ])
    candidate = parameters.data_neural.keys()
    # change when change model_mode
    data_train, train_idx = generate_input_long_history2(parameters.data_neural, 'train',
                                                       candidate=candidate)
    data_test, test_idx = generate_input_long_history2(parameters.data_neural, 'test',
                                                       candidate=candidate)
    # loc_train, tim_train, target_train = run_simple_mod(data_train, train_idx, 'train', parameters.model_mode)
    # loc_test, tim_test, target_test = run_simple_mod(data_test, test_idx, 'test', parameters.model_mode)
    args1 = run_simple_mod(data_train, train_idx, 'train', parameters.model_mode)
    args2 = run_simple_mod(data_test, test_idx, 'test', parameters.model_mode)
    # loc_train, tim_train, target_train, len_train = run_simple_mod(data_train, train_idx, 'train', parameters.model_mode)
    # loc_test, tim_test, target_test, len_test =  run_simple_mod(data_test, test_idx, 'test', parameters.model_mode)
    for i in range(100):

        # model.predict(generator_attn_user(loc_train, tim_train, target_train))
        reduce_lr = ReduceLROnPlateau(monitor= 'loss', factor=parameters.lr_decay,
                              patience=parameters.lr_step, min_lr=parameters.min_lr)
        model.fit(generator_simple(args1), epochs=1)
    result = model.evaluate(generator_simple(args2))
        # model.fit(generator_attn_long(loc_train, tim_train, \
        #     target_train, len_train), steps_per_epoch= 100,epochs=1)
        # result = model.evaluate(generator_attn_long(loc_test, tim_test, target_test,\
        #     len_test))
       
        
    # logits = [[4.0, 3.0, 1.0], [0.0, 5.0, 1.0]]

    # labels = [[1, 0, 0], [0, 10, 0]]
    # print(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))


