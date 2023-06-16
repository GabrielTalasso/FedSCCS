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

#como salvar as ativações sem ser utilizando esse dicionario global?

actv = []
data_path = 'clustering_fl/data'
n_clients = 7

data_perc = 0.01 #percentual de dados que serão compartilhados de cada cliente
x_servidor = pd.DataFrame()

for i in range(n_clients):
  with open(f'{data_path}/client{i+1}.csv', 'rb') as train_file:
    data_client_i = pd.read_csv(train_file).drop('Unnamed: 0', axis = 1) 
    
    #x_servidor = x_servidor.append(data_client_i.sample(int(len(data_client_i)*data_perc)))
    x_servidor = data_client_i.sample(100)

y_servidor =  x_servidor['label'].values
x_servidor.drop('label', axis=1, inplace=True)
# train.drop('subject', axis=1, inplace=True)
# train.drop('trial', axis=1, inplace=True)
x_servidor =  x_servidor.values

#gerando ruido para testar o CKA
noises = []
for i in range(100):

  noises.append((np.random.uniform(0, 255, 28*28)))

#data_test_cka = torch.from_numpy(np.array(noises))
x_servidor = np.array(noises)

def get_layer_outputs(model, layer, input_data, learning_phase=1):
    layer_fn = K.function(model.input, layer.output)
    return layer_fn(input_data)

class NeuralMatch(fl.server.strategy.FedAvg):

  global x_servidor
  global y_servidor

  global actv

  def aggregate_fit(self, server_round, results, failures):
    
    def create_model():
      model = tf.keras.models.Sequential()
      model.add(tf.keras.layers.InputLayer(input_shape=(x_servidor.shape[1],)))
  
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
      lista_modelos['models'].append(parameters_to_ndarrays(parametros_client))

      weights_results.append((parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples))

    lista_last = []
    lista_onetolast = []
    lista_first = []
    lista_second = []

    for w in lista_modelos['models']:

      modelo.set_weights(w)
      modelo.predict(x_servidor) 

      activation_last = get_layer_outputs(modelo, modelo.layers[-1], x_servidor, 0)
      activation_onetolast = get_layer_outputs(modelo, modelo.layers[-2], x_servidor, 0)

      activation_first = get_layer_outputs(modelo, modelo.layers[0], x_servidor, 0) 
      activation_second = get_layer_outputs(modelo, modelo.layers[1], x_servidor, 0)

      lista_last.append(activation_last)
      lista_onetolast.append(activation_onetolast)
      lista_first.append(activation_first)
      lista_second.append(activation_second)

    lista_modelos['actv_last'] = lista_last
    lista_modelos['actv_onetolast'] = lista_onetolast
    lista_modelos['actv_first'] = lista_first
    lista_modelos['actv_second'] = lista_second

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
                              'actv_last' : actv[0]['actv_last'] ,
                              'actv_onetolast': actv[0]['actv_onetolast'],
                              'actv_first': actv[0]['actv_first'],
                              'actv_second': actv[0]['actv_second']}


        return loss_aggregated, metrics_aggregated