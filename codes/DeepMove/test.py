# coding: utf-8

from datetime import datetime
import sys
import operator
from python_utils import *


from model import TrajPreSimple, TrajPreAttnAvgLongUser, TrajPreLocalAttnLong
from train import run_simple, RnnParameterData, generate_input_history, markov, \
    generate_input_long_history, generate_input_long_history2


import torch
import torch.nn as nn
import torch.optim as optim

import os
import json
import time
import argparse
import numpy as np
from json import encoder
from scipy.special import softmax


class RNNNumpy:
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
        fc1 = np.random.uniform(0, 1, (self.word_dim, 1))
        x = np.reshape(x, (1, x.shape[0]))
        input_x = np.dot(fc1, x)
        # print(fc.shape)
        # print(x.shape)

        for t in np.arange(T):
            s[t] = np.tanh(self.U.dot(input_x[:, t]) + self.W.dot(s[t - 1]))

        # s[:-1] = np.tanh(self.U.dot(input_x)) + s[1:].dot(self.W)
        # print(s.shape)
        o = softmax(np.dot(s, self.V.transpose()), axis=1)

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
        fc = np.random.uniform(
            0, 1, (self.total_vocabulary_size, self.word_dim))
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
        dLdU = np.dot(dLdU_mod, fc)
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


# Outer SGD Loop
# - model: The RNN model instance
# - X_train: The training data set
# - y_train: The training data labels
# - learning_rate: Initial learning rate for SGD
# - nepoch: Number of times to iterate through the complete dataset
# - evaluate_loss_after: Evaluate the loss after this many epochs
def train_with_sgd(model, X_train, y_train, learning_rate=0.01, nepoch=100, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(1, nepoch):
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss, accuracy = model.calculate_loss(X_train, y_train)
            # print("accuracy={0:.10f}%".format(100 * accuracy))
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("{0}: After num_examples_seen={1} epoch={2}: loss={3: f} accuracy={4: .5f}%".format
                  (time, num_examples_seen, epoch, loss, 100 * accuracy))
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5

            else:
                learning_rate = learning_rate * 1.1
            print("Setting learning rate to %f" % learning_rate)
            # sys.stdout.flush()
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1


if __name__ == '__main__':
    # this data is produced by script data_preprocess
    # X_train array of list, a list represent a sentence, y_train is the origin sentence shifted to right by 1 word.

    # sentence[0]: [12,31,3234,42,53]
    #X_train[0]: [12,31,3234,42]
    #y_train[1]: [31,3234,42,53]

    # index_to_word = np.load('data/index_to_word.npy')
    # word_to_index = np.load('/Users/iphone13.5/Desktop')
    X_train = np.load('/Users/iphone13.5/Desktop/x_test.npy').astype(int)
    y_train = np.load('/Users/iphone13.5/Desktop/x_test.npy').astype(int)
    listx = list()
    listy = list()
    for i in range(1, 150):
        listx.append(X_train[i, : 750 - i])
        listy.append(y_train[i, : 750 - i])
    # for i in range(147, 700):
    #     listx.append(X_train[i])
    #     listy.append(y_train[i])
    # print(X_train)

    # print(X_train.shape)

    vocabulary_size = 60
    total_vocabulary_size = X_train.shape[1]
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
    model = RNNNumpy(vocabulary_size, total_vocabulary_size)
    losses = train_with_sgd(
        model, listx, listy, nepoch=200, evaluate_loss_after=1)
