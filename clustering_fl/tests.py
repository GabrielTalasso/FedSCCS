import pandas as pd
import pickle
import tensorflow as tf
import numpy as np
import pandas as pd



n_clients = 7
data_path = 'data/MNIST'

with open(f'./{data_path}/0/classes_per_client_10/alpha_0.1/idx_train_0.pickle', 'rb') as file:
    f  = pickle.load(file)
#data/MNIST/0/classes_per_client_10/alpha_0.1/idx_test_0.pickle
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#
#print(f)
x_train[f]