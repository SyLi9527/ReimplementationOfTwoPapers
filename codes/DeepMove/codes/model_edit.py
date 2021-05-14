import numpy as np
from python_utils import *
import operator
import sys
from datetime import datetime
from scipy.special import softmax
import cupy as cp


class TrajPreSimple1:
    def __init__(self, loc_dim, tim_dim, hidden_dim=55, bptt_truncate=4):
        # Assign instance variables
        self.loc_dim = loc_dim
        self.tim_dim = tim_dim
        # self.word_dim = self.loc_dim + self.tim_dim
        self.word_dim = self.loc_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        # s_t = T.tanh(U[:,x_t] + W.dot(s_t1_prev))

        self.U = np.random.uniform(-np.sqrt(1. / self.word_dim),
                                   np.sqrt(1. / self.word_dim), (hidden_dim, self.word_dim))
        self.V = np.random.uniform(-np.sqrt(1. / hidden_dim),
                                   np.sqrt(1. / hidden_dim), (self.word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1. / hidden_dim),
                                   np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))

    def forward_propagation(self, loc, tim):
        # The total number of time steps
        x = np.append(loc, tim)
        print(loc.shape)
        print(x.shape)
        T = len(x)

        # print(x.shape)
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        s = np.zeros((T + 1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)
        # The outputs at each time step. Again, we save them for later.
        o = np.zeros((T, self.word_dim))
        # For each time step...
        for t in np.arange(T):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t - 1]))
            o[t] = softmax(self.V.dot(s[t]))
        # print(o[t])
        # print(s.shape)
        return [o, s]

    def predict(self, o, y):
        # Perform forward propagation and return index of the highest score
        o_index = np.argmax(o, axis=1)
        # print(o_index.shape)
        # print(o.shape)
        # print(o_index.shape)
        # print(y.shape)
        return sum(o_index[i] == y[i] for i in range(len(y)))

    def calculate_total_loss(self, loc, tim, y):
        L = 0
        sumA = 0
        # For each sentence...

        for i in np.arange(len(y)):
            o, s, x = self.forward_propagation(loc[i], tim[i])
            # print(y[i].shape)
            sumA += self.predict(o, y[i])
            # We only care about our prediction of the "correct" words
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            # print(len(y[i]))
            # Add to the loss based on how off we were
            L += -1 * np.sum(np.log(correct_word_predictions))
        return L, sumA

    def calculate_loss(self, loc, tim, y):
        # Divide the total loss by the number of training examples
        N = sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(loc, tim, y)[0] / N, self.calculate_total_loss(loc, tim, y)[1] / N

    def bptt(self, loc, tim, y):
        T = len(y)
        # Perform forward propagation
        o, s, input_x = self.forward_propagation(loc, tim)
        x = np.append(loc, tim)

        # We accumulate the gradients in these variables
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1.
        # For each output backwards...
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)
            # Initial delta calculation
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            # Backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in np.arange(max(0, t - self.bptt_truncate), t + 1)[::-1]:
                # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
                dLdW += np.outer(delta_t, s[bptt_step - 1])
                dLdU[:, x[bptt_step]] += delta_t
                # Update delta for next step
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step - 1] ** 2)
        return [dLdU, dLdV, dLdW]

    def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
        # Calculate the gradients using backpropagation. We want to checker if these are correct.
        bptt_gradients = model.bptt(x, y)
        # List of all parameters we want to check.
        model_parameters = ['U', 'V', 'W']
        # Gradient check for each parameter
        for pidx, pname in enumerate(model_parameters):
            # Get the actual parameter value from the mode, e.g. model.W
            parameter = operator.attrgetter(pname)(self)
            print
            "Performing gradient check for parameter %s with size %d." % (
                pname, np.prod(parameter.shape))
            # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
            it = np.nditer(parameter, flags=[
                           'multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                # Save the original value so we can reset it later
                original_value = parameter[ix]
                # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
                parameter[ix] = original_value + h
                gradplus = model.calculate_total_loss([x], [y])[0]
                parameter[ix] = original_value - h
                gradminus = model.calculate_total_loss([x], [y])[0]
                estimated_gradient = (gradplus - gradminus) / (2 * h)
                # Reset parameter to original value
                parameter[ix] = original_value
                # The gradient for this parameter calculated using backpropagation
                backprop_gradient = bptt_gradients[pidx][ix]
                # calculate The relative error: (|x - y|/(|x| + |y|))
                relative_error = np.abs(backprop_gradient - estimated_gradient) / (
                    np.abs(backprop_gradient) + np.abs(estimated_gradient))
                # If the error is to large fail the gradient check
                if relative_error > error_threshold:
                    print
                    "Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix)
                    print
                    "+h Loss: %f" % gradplus
                    print
                    "-h Loss: %f" % gradminus
                    print
                    "Estimated_gradient: %f" % estimated_gradient
                    print
                    "Backpropagation gradient: %f" % backprop_gradient
                    print
                    "Relative Error: %f" % relative_error
                    return
                it.iternext()
            print
            "Gradient check for parameter %s passed." % (pname)

    # Performs one step of SGD.

    def sgd_step(self, loc, tim, y, learning_rate):
        # Calculate the gradients
        dLdU, dLdV, dLdW = self.bptt(loc, tim, y)
        # Change parameters according to gradients and learning rate
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW


class TrajPreSimple2:
    def __init__(self, loc_dim, loc_emb_size, tim_emb_size, hidden_dim=35, bptt_truncate=4):
        # Assign instance variables
        self.input_dim = loc_emb_size + tim_emb_size
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.total_location_size = loc_dim
        # Randomly initialize the network parameters
        # s_t = T.tanh(U[:,x_t] + W.dot(s_t1_prev))

        self.U = np.random.uniform(-np.sqrt(1. / self.input_dim),
                                   np.sqrt(1. / self.input_dim), (self.hidden_dim, self.input_dim))
        self.V = np.random.uniform(-np.sqrt(1. / self.hidden_dim),
                                   np.sqrt(1. / self.hidden_dim), (self.total_location_size, self.hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1. / self.hidden_dim),
                                   np.sqrt(1. / self.hidden_dim), (self.hidden_dim, self.hidden_dim))
        # self.fc1 = np.random.uniform(0, 1, (self.input_dim, 1))
        self.fc_loc = np.random.normal(0, 1, (loc_emb_size, 1))
        self.fc_tim = np.random.normal(0, 1, (tim_emb_size, 1))
        self.fc2 = np.random.uniform(-np.sqrt(1. / self.total_location_size),
                                     np.sqrt(1. / self.total_location_size), (self.total_location_size, self.input_dim))

    def forward_propagation(self, loc, tim):
        # The total number of time steps

        T = len(loc)
        # print(x.shape)
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        s = np.zeros((T + 1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)
        # The outputs at each time step. Again, we save them for later.
        o = np.zeros((T, self.total_location_size))

        # For each time step...
        # for t in np.arange(T):
        # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
        # print(x[t].shape[0])

        loc = np.reshape(loc, (1, loc.shape[0]))
        tim = np.reshape(tim, (1, tim.shape[0]))
        input_loc = np.dot(self.fc_loc, loc)
        input_tim = np.dot(self.fc_tim, tim)
        input_x = np.append(input_loc, input_tim, axis=0)
        # print(fc.shape)
        # print(x.shape)

        for t in np.arange(T):
            s[t] = np.tanh(self.U.dot(input_x[:, t]) + self.W.dot(s[t - 1]))
            o[t] = softmax(self.V.dot(s[t]))

        # s[:-1] = np.tanh(self.U.dot(input_x)) + s[1:].dot(self.W)
        # print(s.shape)
        # o = softmax(np.dot(s, self.V.transpose()), axis=1)

        # for i in o:
        #     if i == 0:
        #         print("0 exists")
        #         break
        return [o, s, input_x]

    def predict(self, o, y):
        # Perform forward propagation and return index of the highest score
        o_index = np.argmax(o, axis=1)
        return sum(o_index[i] == y[i] for i in range(len(y)))

    def calculate_total_loss(self, loc, tim, y):
        L = 0
        sumA = 0
        # For each sentence...

        for i in np.arange(len(y)):
            o, s, input_x = self.forward_propagation(loc[i], tim[i])
            # print(y[i].shape)
            sumA += self.predict(o, y[i])
            # We only care about our prediction of the "correct" words
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            # print(correct_word_predictions)
            # print(len(y[i]))
            # Add to the loss based on how off we were
            L += -1 * cp.sum(np.log(correct_word_predictions))
        return L, sumA

    def calculate_loss(self, loc, tim, y):
        # Divide the total loss by the number of training examples
        N = sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(loc, tim, y)[0] / N, self.calculate_total_loss(loc, tim, y)[1] / N

    def bptt(self, loc, tim, y):
        # print(y.shape)
        T = len(y)
        # Perform forward propagation
        o, s, input_x = self.forward_propagation(loc, tim)

        # We accumulate the gradients in these variables

        # print(self.U.shape)
        # print(self.V.shape)
        # print(self.W.shape)

        dLdU = np.zeros(self.U.shape)

        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1.
        # For each output backwards...
        for t in np.arange(T)[::-1]:
            # print(delta_o[t].shape)
            # print(s[t].shape)
            dLdV += np.outer(delta_o[t], s[t].T)
            # Initial delta calculation
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))

            # Backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in np.arange(max(0, t - self.bptt_truncate), t + 1)[::-1]:
                # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
                dLdW += np.outer(delta_t, s[bptt_step - 1])
                # dLdU_mod[:, x[bptt_step]] += delta_t
                dLdU += np.outer(delta_t, input_x[:, bptt_step])
                # Update delta for next step
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step - 1] ** 2)
        # dLdU = np.dot(dLdU_mod, self.fc2)
        return [dLdU, dLdV, dLdW]

    def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
        # Calculate the gradients using backpropagation. We want to checker if these are correct.
        bptt_gradients = model.bptt(x, y)
        # List of all parameters we want to check.
        model_parameters = ['U', 'V', 'W']
        # Gradient check for each parameter
        for pidx, pname in enumerate(model_parameters):
            # Get the actual parameter value from the mode, e.g. model.W
            parameter = operator.attrgetter(pname)(self)
            print
            "Performing gradient check for parameter %s with size %d." % (
                pname, np.prod(parameter.shape))
            # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
            it = np.nditer(parameter, flags=[
                           'multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                # Save the original value so we can reset it later
                original_value = parameter[ix]
                # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
                parameter[ix] = original_value + h
                gradplus = model.calculate_total_loss([x], [y])[0]
                parameter[ix] = original_value - h
                gradminus = model.calculate_total_loss([x], [y])[0]
                estimated_gradient = (gradplus - gradminus) / (2 * h)
                # Reset parameter to original value
                parameter[ix] = original_value
                # The gradient for this parameter calculated using backpropagation
                backprop_gradient = bptt_gradients[pidx][ix]
                # calculate The relative error: (|x - y|/(|x| + |y|))
                relative_error = np.abs(backprop_gradient - estimated_gradient) / (
                    np.abs(backprop_gradient) + np.abs(estimated_gradient))
                # If the error is to large fail the gradient check
                if relative_error > error_threshold:
                    print
                    "Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix)
                    print
                    "+h Loss: %f" % gradplus
                    print
                    "-h Loss: %f" % gradminus
                    print
                    "Estimated_gradient: %f" % estimated_gradient
                    print
                    "Backpropagation gradient: %f" % backprop_gradient
                    print
                    "Relative Error: %f" % relative_error
                    return
                it.iternext()
            print
            "Gradient check for parameter %s passed." % (pname)

    # Performs one step of SGD.
    def sgd_step(self, loc, tim, y, learning_rate):
        # Calculate the gradients
        dLdU, dLdV, dLdW = self.bptt(loc, tim, y)
        # Change parameters according to gradients and learning rate
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW


class TrajPreAttnAvgLongUser:
    def __init__(self):
        pass


class TrajPreLocalAttnLong:
    def __init__(self):
        pass


if __name__ == '__main__':
    # this data is produced by script data_preprocess
    # X_train array of list, a list represent a sentence, y_train is the origin sentence shifted to right by 1 word.

    # sentence[0]: [12,31,3234,42,53]
    # X_train[0]: [12,31,3234,42]
    #y_train[1]: [31,3234,42,53]

    # index_to_word = np.load('data/index_to_word.npy')
    # word_to_index = np.load('/Users/iphone13.5/Desktop')
    X_train = np.load('/Users/iphone13.5/Desktop/x_test.npy').astype(int)
    y_train = np.load('/Users/iphone13.5/Desktop/x_test.npy').astype(int)
    # print(y_train.shape)
    listx = list()
    listy = list()
    for i in range(1, 150):
        listx.append(X_train[i, : 750 - i])
        listy.append(y_train[i, 1: 751 - i])
    # for i in range(147, 700):
    #     listx.append(X_train[i])
    #     listy.append(y_train[i])
    # print(X_train)

    # print(X_train.shape)

    vocabulary_size = X_train.shape[1]

    print("vocabulary size: %d" % vocabulary_size)
    # model = RNNNumpy(vocabulary_size)
    # # test cross entropy loss
    # print("Expected Loss for random predictions: %f" % np.log(vocabulary_size))
    # print("Actual loss: %f" % model.calculate_loss(
    #     listx, listy))

    # # gradient checking
    # # To avoid performing millions of expensive calculations we use a smaller vocabulary size for checking.
    # grad_check_vocab_size = 100
    # np.random.seed(10)
    # model = RNNNumpy(grad_check_vocab_size, 10, bptt_truncate=1000)
    # model.gradient_check([0, 1, 2, 3], [1, 2, 3, 4])

    # train
    np.random.seed(11)
    # Train on a small subset of the data to see what happens
    # for list1 in listy:
    #     for i in list1:
    #         if i == 0:
    #             print('yes')

    model = TrajPreSimple1(500, total_location_size=vocabulary_size)
    losses = train_simple(
        model, listx, listy, nepoch=20, evaluate_loss_after=1)
