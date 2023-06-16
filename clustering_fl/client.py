import flwr as fl
import pickle
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.special import rel_entr
from scipy.stats import wasserstein_distance

import tensorflow as tf
import numpy as np
import random

## onde colocar o datapath e x_servidor?? (arrumar tambem no servidor)
data_path = 'clustering_fl/data'
n_clients = 7


class ClientBase(fl.client.NumPyClient):

	def __init__(self, cid):

		self.cid = cid
		self.x_train, self.y_train, self.x_test, self.y_test = self.load_data()
		self.model     = self.create_model()
		self.round = 0
		print('++++++++++++++++++++++++++')

	def load_data(self):
		with open(f'{data_path}/client{self.cid+1}.csv', 'rb') as train_file:
			train = pd.read_csv(train_file).drop('Unnamed: 0', axis = 1) 
	    
		with open(f'{data_path}/mnist_test.csv', 'rb') as test_file: 	 
			test = pd.read_csv(test_file, dtype = np.float32)
			test = test.rename({'7': 'label'}, axis = 1)
	        
		y_train = train['label'].values
		train.drop('label', axis=1, inplace=True)
		# train.drop('subject', axis=1, inplace=True)
		# train.drop('trial', axis=1, inplace=True)
		x_train = train.values

		y_test = test['label'].values
		test.drop('label', axis=1, inplace=True)
		# test.drop('subject', axis=1, inplace=True)
		# test.drop('trial', axis=1, inplace=True)
		x_test = test.values
	    
		return x_train, y_train, x_test, y_test

	def create_model(self):
		model = tf.keras.models.Sequential()
		model.add(tf.keras.layers.InputLayer(input_shape=(self.x_train.shape[1],)))

		model.add(tf.keras.layers.Dense(128, activation='relu'))
	
		model.add(tf.keras.layers.Dense(128, activation='tanh'))
	
		model.add(tf.keras.layers.Dense(128, activation='elu'))
	
		model.add(tf.keras.layers.Dense(128, activation='relu',))
	
		model.add(tf.keras.layers.Dense(10, activation='softmax'))

		model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
		return model

	def get_parameters(self, config):
		return self.model.get_weights()

	def fit(self, parameters, config):
		
		print(config)

		#config recebe parametros do servidor
		self.model.set_weights(parameters)
		h = self.model.fit(self.x_train, self.y_train, 
		                                validation_data = (self.x_test, self.y_test),
																		verbose=1, epochs=1)

	
		# '''
		# adicionar metodos
		# metricas
		# ativações
		# modelos
		# '''

		msg2server = {
				"cliente_id": self.cid,
				# "ativacoes" : var_ativacoes
		}
		self.round += 1
		with open('acc.csv', 'a') as arquivo:

			arquivo.write(f"{self.round}, {self.cid}, {np.mean(h.history['accuracy'])}, {np.mean(h.history['loss'])}\n")
	 


		#acc = pd.read_csv('/content/acc.csv')		
		#acc = acc.append({'cid':self.cid, 
		#            'acc':accuracy}, ignore_index = True)
		#acc[['cid', 'acc']].to_csv('acc.csv')

		#config usado para passarar parametros para o servidor
		return self.model.get_weights(), len(self.x_train), msg2server


	def evaluate(self, parameters, config):
		self.model.set_weights(parameters)

		loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
	
	
		return loss, len(self.x_test), {"accuracy" : accuracy}