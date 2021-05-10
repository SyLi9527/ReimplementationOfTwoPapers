import numpy as np
from python_utils import *
import operator
import sys
from datetime import datetime
from scipy.special import softmax


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
            o, s = self.forward_propagation(loc[i], tim[i])
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
        o, s = self.forward_propagation(loc, tim)
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
    def __init__(self, word_dim, total_vocabulary_size, hidden_dim=55, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.total_vocabulary_size = total_vocabulary_size
        # Randomly initialize the network parameters
        # s_t = T.tanh(U[:,x_t] + W.dot(s_t1_prev))

        self.U = np.random.uniform(-np.sqrt(1. / word_dim),
                                   np.sqrt(1. / word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1. / hidden_dim),
                                   np.sqrt(1. / hidden_dim), (total_vocabulary_size, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1. / hidden_dim),
                                   np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))
        self.fc1 = np.random.uniform(0, 1, (self.word_dim, 1))
        self.fc2 = np.random.uniform(
            0, 1, (self.total_vocabulary_size, self.word_dim))

    def forward_propagation(self, x):
        # The total number of time steps
        T = len(x)
        # print(x.shape)
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        s = np.zeros((T + 1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)
        # The outputs at each time step. Again, we save them for later.
        o = np.zeros((T, self.total_vocabulary_size))

        # For each time step...
        # for t in np.arange(T):
        # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
        # print(x[t].shape[0])

        x = np.reshape(x, (1, x.shape[0]))
        input_x = np.dot(self.fc1, x)
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
        return [o, s]

    def predict(self, o, y):
        # Perform forward propagation and return index of the highest score
        o_index = np.argmax(o, axis=1)
        return sum(o_index[i] == y[i] for i in range(len(y)))

    def calculate_total_loss(self, x_list, y_list):
        L = 0
        sumA = 0
        # For each sentence...

        for i in np.arange(len(y_list)):
            o, s = self.forward_propagation(x_list[i])
            # print(y[i].shape)
            sumA += self.predict(o, y_list[i])
            # We only care about our prediction of the "correct" words
            correct_word_predictions = o[np.arange(len(y_list[i])), y_list[i]]
            # print(correct_word_predictions)
            # print(len(y[i]))
            # Add to the loss based on how off we were
            L += -1 * np.sum(np.log(correct_word_predictions))
        return L, sumA

    def calculate_loss(self, x_list, y_list):
        # Divide the total loss by the number of training examples
        N = sum((len(y_i) for y_i in y_list))
        return self.calculate_total_loss(x_list, y_list)[0] / N, self.calculate_total_loss(x_list, y_list)[1] / N

    def bptt(self, x, y):
        # print(y.shape)
        T = len(y)
        # Perform forward propagation
        o, s = self.forward_propagation(x)
        # We accumulate the gradients in these variables

        # print(self.U.shape)
        # print(self.V.shape)
        # print(self.W.shape)

        dLdU = np.zeros(self.U.shape)
        dLdU_mod = np.zeros((self.hidden_dim, self.total_vocabulary_size))

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
                dLdU_mod[:, x[bptt_step]] += delta_t
                # Update delta for next step
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step - 1] ** 2)
        dLdU = np.dot(dLdU_mod, self.fc2)
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
    def sgd_step(self, x, y, learning_rate):
        # Calculate the gradients
        dLdU, dLdV, dLdW = self.bptt(x, y)
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
