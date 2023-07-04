"""## 2. Desenvolvendo Classe Servidor"""

import flwr as fl
import tensorflow as tf
import torch
from logging import WARNING
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.strategy.strategy import Strategy

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from keras.layers import Input, Dense, Activation
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K

import pandas as pd
from server_utils import *

#como salvar as ativações sem ser utilizando esse dicionario global?

actv = []
data_path = './data'
n_clients = 10
clustering = True
K = 5

(x_servidor, _), (_, _) = tf.keras.datasets.mnist.load_data()
x_servidor = x_servidor[list(np.random.random_integers(1,6000, 100))]
x_servidor = x_servidor.reshape(x_servidor.shape[0] , 28*28)
#
#data_perc = 0.01 #percentual de dados que serão compartilhados de cada cliente
#x_servidor = pd.DataFrame()
#
#for i in range(n_clients):
#  with open(f'{data_path}/client{i+1}.csv', 'rb') as train_file:
#    data_client_i = pd.read_csv(train_file).drop('Unnamed: 0', axis = 1) 
#    x_servidor = pd.concat([data_client_i.sample(int(len(data_client_i)*data_perc)), x_servidor],
#                           ignore_index = True)
#
#y_servidor =  x_servidor['label'].values
#x_servidor.drop('label', axis=1, inplace=True)
# train.drop('subject', axis=1, inplace=True)
# train.drop('trial', axis=1, inplace=True)
#x_servidor =  x_servidor.values

def get_layer_outputs(model, layer, input_data, learning_phase=1):
    layer_fn = K.function(model.input, layer.output)
    return layer_fn(input_data)

modelos = []
idx = list(np.zeros(n_clients))

class NeuralMatch(fl.server.strategy.FedAvg):

  global x_servidor
  global y_servidor
  global modelos
  global actv
  global idx
  idx = list(np.zeros(n_clients))

  def aggregate_fit(self, server_round, results, failures):
    
    def create_model():
      model = tf.keras.models.Sequential()
      model.add(tf.keras.layers.Flatten(input_shape=(784, )))
  
      model.add(tf.keras.layers.Dense(128, activation='relu'))
  
      model.add(tf.keras.layers.Dense(128, activation='tanh'))
  
      model.add(tf.keras.layers.Dense(128, activation='elu'))

      model.add(tf.keras.layers.Dense(128, activation='relu',))
  
      model.add(tf.keras.layers.Dense(10, activation='softmax'))

      return model

    modelo = create_model()

    """Aggregate fit results using weighted average."""
    lista_modelos = {'cids': [], 'models' : []}


    # Convert results
    weights_results = []
    for _, fit_res in results:

      client_id = str(fit_res.metrics['cliente_id'])
      parametros_client = fit_res.parameters

      #salvando os modelos (pesos)
      lista_modelos['cids'].append(client_id)

      #idx_cluster = idx[i]
      #lista_modelos['models'][idx_cluster].append(parameters_to_ndarrays(parametros_client))

      lista_modelos['models'].append(parameters_to_ndarrays(parametros_client))
      
      weights_results.append((parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples))

    lista_last = []

    for w in lista_modelos['models']:

      modelo.set_weights(w)
      modelo.predict(x_servidor) 

      activation_last = get_layer_outputs(modelo, modelo.layers[-1], x_servidor, 0)
      lista_last.append(activation_last)

    lista_modelos['actv_last'] = lista_last

    actvs = lista_last

    matrix = np.zeros((len(actvs), len(actvs)))

    for i , a in enumerate(actvs):
      for j, b in enumerate(actvs):

        x = int(lista_modelos['cids'][i])
        y = int(lista_modelos['cids'][j])

        matrix[x][y] = cka(a, b)

    if clustering:
       
       if server_round%2 == 0:
       
        idx = server_Hclusters(matrix, K, plot_dendrogram=True)

        
    #criar um for para cada cluster ter um modelo   
    parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

    actv.append(lista_modelos)

    metrics_aggregated = {}

    return parameters_aggregated, metrics_aggregated

  def aggregate_evaluate(
        self,
        server_round,
        results,
        failures,
    ):
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided

        #print(actv[0].keys())

        metrics_aggregated = {'str':server_round, 
                              'cids' : actv[0]['cids'],
                              'actv_last' : actv[0]['actv_last']}


        return loss_aggregated, metrics_aggregated