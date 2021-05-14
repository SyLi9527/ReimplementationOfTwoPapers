from numpy.core.numeric import outer
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop
# import keras
import numpy as np
import os


loc_input = Input(shape=(1,), dtype=tf.int16, ragged=True)
x = Embedding(100,10)
y = x(loc_input)

x.get_weights()
print(y)