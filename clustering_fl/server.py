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

from dataset_utils import ManageDatasets
from model_definition import ModelCreation

from sys import getsizeof

n_clients = 25

def get_layer_outputs(model, layer, input_data, learning_phase=1):
    layer_fn = K.function(model.input, layer.output)
    return layer_fn(input_data)

idx = list(np.zeros(n_clients))

class NeuralMatch(fl.server.strategy.FedAvg):

  def __init__(self, model_name, n_clusters, n_clients, clustering, clustering_round, dataset, fraction_fit):

    self.model_name = model_name
    self.n_clusters = n_clusters
    self.n_clients = n_clients
    self.clustering = clustering
    self.clustering_round = clustering_round
    self.dataset = dataset

    super().__init__(fraction_fit=1, 
		    min_available_clients=self.n_clients, 
		    min_fit_clients=self.n_clients, 
		    min_evaluate_clients=self.n_clients)
    
    if dataset == 'MNIST':
      (x_servidor, _), (_, _) = tf.keras.datasets.mnist.load_data()
      x_servidor = x_servidor[list(np.random.random_integers(1,60000, 1000))]
      self.x_servidor = x_servidor.reshape(x_servidor.shape[0] , 28*28) 

    if dataset == 'CIFAR10':
      (x_servidor, _), (_, _) = tf.keras.datasets.cifar10.load_data()
      self.x_servidor = x_servidor[list(np.random.random_integers(1,50000, 1000))]
      #self.x_servidor = x_servidor.reshape(x_servidor.shape[0] , 32*32) 
        
  global idx

  def aggregate_fit(self, server_round, results, failures):
    global idx  

    def create_model(self):
      input_shape = self.x_servidor.shape

      if self.model_name == 'DNN':
        return ModelCreation().create_DNN(input_shape, 10)

      elif self.model_name == 'CNN':
        return ModelCreation().create_CNN(input_shape, 10)

    modelo = create_model(self)

    """Aggregate fit results using weighted average."""
    lista_modelos = {'cids': [], 'models' : {}, 'cluster': []}

    # Convert results
    weights_results = {}
    lista_last = []
    for _, fit_res in results:

      client_id = str(fit_res.metrics['cliente_id'])
      parametros_client = fit_res.parameters

      lista_modelos['cids'].append(client_id)

      idx_cluster = idx[int(client_id)]
      #print(idx)
      if str(idx_cluster) not in lista_modelos['models'].keys(): 
        lista_modelos['models'][str(idx_cluster)] = []
      lista_modelos['models'][str(idx_cluster)].append(parameters_to_ndarrays(parametros_client))
      
      if str(idx_cluster) not in weights_results.keys():
          weights_results[str(idx_cluster)] = []
      weights_results[str(idx_cluster)].append((parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples))


      w = lista_modelos['models'][str(idx_cluster)][-1]#
      modelo.set_weights(w)#
      modelo.predict(self.x_servidor)# 

      activation_last = get_layer_outputs(modelo, modelo.layers[-2], self.x_servidor, 0)#
      lista_last.append(activation_last)#

    lista_modelos['actv_last'] = lista_last.copy()

    actvs = lista_last.copy()

    matrix = np.zeros((len(actvs), len(actvs)))

    for i , a in enumerate(actvs):
      for j, b in enumerate(actvs):

        x = int(lista_modelos['cids'][i])
        y = int(lista_modelos['cids'][j])

        matrix[x][y] = cka(a, b)

    #print(matrix)

    if self.clustering:
      if (server_round == self.clustering_round-1) or (server_round == self.clustering_round) or (server_round%50 == 0):
        idx = server_Hclusters(matrix, self.n_clusters, plot_dendrogram=False)
        #idx = server_Hclusters2(matrix, plot_dendrogram=True)

        with open(f'results/clusters_{self.dataset}_{self.n_clients}clients_{self.n_clusters}clusters.txt', 'a') as arq:
          arq.write(f"{idx} - round{server_round}\n")
        
    #criar um for para cada cluster ter um modelo
    parameters_aggregated = {}
    for idx_cluster in weights_results.keys():   
      parameters_aggregated[idx_cluster] = ndarrays_to_parameters(aggregate(weights_results[idx_cluster]))

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
        metrics_aggregated = {}


        return loss_aggregated, metrics_aggregated
  
  def configure_fit(
        self, server_round, parameters, client_manager):
        """Configure the next round of training."""
        config = {}
        global idx 
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        # Return client/config pairs

        if server_round == 1:
           fit_ins = FitIns(parameters, config)
           return [(client, fit_ins) for client in clients]
        
        elif server_round <= self.clustering_round:
           fit_ins = FitIns(parameters['0.0'], config)
           return [(client, fit_ins) for client in clients]
        
        else:
          return [(client, FitIns(parameters[str(idx[int(client.cid)])], config)) for client in clients]
  
  def configure_evaluate(
      self, server_round, parameters, client_manager):
      """Configure the next round of evaluation."""
      # Do not configure federated evaluation if fraction eval is 0.
      if self.fraction_evaluate == 0.0:
          return []
      # Parameters and config
      config = {}
      global idx 
      if self.on_evaluate_config_fn is not None:
          # Custom evaluation config function provided
          config = self.on_evaluate_config_fn(server_round)

      # Sample clients
      sample_size, min_num_clients = self.num_evaluation_clients(
          client_manager.num_available()
      )
      clients = client_manager.sample(
          num_clients=sample_size, min_num_clients=min_num_clients
      )
      if server_round == 1:
        evaluate_ins = EvaluateIns(parameters['0.0'], config)
        return [(client, evaluate_ins) for client in clients]
      
      elif server_round == self.clustering_round-1:
        evaluate_ins = EvaluateIns(parameters['0.0'], config)
        return [(client, evaluate_ins) for client in clients]      
      else:
        return [(client, EvaluateIns(parameters[str(idx[int(client.cid)])], config)) for client in clients]