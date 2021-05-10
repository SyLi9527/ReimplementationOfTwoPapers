import numpy as np

# class RnnSimple(object):
#     def __init__(self, x, a0, paras):
#         super(RnnSimple, self).__init__()


def rnn_cell_forward(xt, a_prev, parameters):
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]
    a_next = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, xt) + ba)
    yt_pred = np.softmax(np.dot(Wya, a_next) + by)
    cache = (a_next, a_prev, xt, parameters)
    return a_next, yt_pred, cache


def rnn_forward(x, a0, hidden_size, parameters):

    caches = []
    x = x.transpose()
    # Retrieve dimensions from shapes of x and Wy
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape

    ### START CODE HERE ###

    # initialize "a" and "y" with zeros (≈2 lines)
    a = np.zeros((hidden_size, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))

    # Initialize a_next (≈1 line)
    a_next = a0

    # loop over all time-steps
    for t in range(T_x):
        # Update next hidden state, compute the prediction, get the cache (≈1 line)
        a_next, yt_pred, cache = rnn_cell_forward(
            x[:, :, t], a_next, parameters)
        # Save the value of the new "next" hidden state in a (≈1 line)
        a[:, :, t] = a_next
        # Save the value of the prediction in y (≈1 line)
        y_pred[:, :, t] = yt_pred

        # Append "cache" to "caches" (≈1 line)
        caches.append(cache)

    ### END CODE HERE ###

    # store values needed for backward propagation in cache
    caches = (caches, x)

    return a, y_pred, caches


def bptt(x, targets, a0, hidden_size, parameters):

    T = len(targets)

    # Perform forward propagation
    s, o, = rnn_forward(x, a0, hidden_size, parameters)

    # We accumulate the gradients in these variables

    dLdU = np.zeros(parameters["Wax"].shape)
    dLdV = np.zeros(parameters["Wya"].shape)
    dLdW = np.zeros(parameters["Waa"].shape)

    delta_o = o

    delta_o[np.arange(T), targets] -= 1.

    # For each output backwards...

    for t in np.arange(T)[::-1]:

        dLdV += np.outer(delta_o[t], s[t].T)

        # Initial delta calculation: dL/dz

        delta_t = parameters["Wya"].T.dot(delta_o[t]) * (1 - (s[t] ** 2))

        # Backpropagation through time (for at most self.bptt_truncate steps)

        for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:

            # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)

            # Add to gradients at each previous step

            dLdW += np.outer(delta_t, s[bptt_step-1])

            dLdU[:, x[bptt_step]] += delta_t

            # Update delta for next step dL/dz at t-1

            delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)

    return [dLdU, dLdV, dLdW]


def rnn_cell_backward(da_next, cache):

    # Retrieve values from cache
    (a_next, a_prev, xt, parameters) = cache

    # Retrieve values from parameters
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    ba = parameters["ba"]

    ### START CODE HERE ###
    # compute the gradient of tanh with respect to a_next (≈1 line)
    dtanh = (1 - a_next**2) * da_next

    # compute the gradient of the loss with respect to Wax (≈2 lines)
    dxt = np.dot(Wax.T, dtanh)
    dWax = np.dot(dtanh, xt.T)

    # compute the gradient with respect to Waa (≈2 lines)
    da_prev = np.dot(Waa.T, dtanh)
    dWaa = np.dot(dtanh, a_prev.T)

    # compute the gradient with respect to b (≈1 line)
    dba = np.sum(dtanh, 1, keepdims=True)

    gradients = {"dxt": dxt, "da_prev": da_prev,
                 "dWax": dWax, "dWaa": dWaa, "dba": dba}

    return gradients
