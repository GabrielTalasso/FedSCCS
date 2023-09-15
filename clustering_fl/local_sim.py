from client import ClientBase
from server import NeuralMatch
import pickle
import flwr as fl
import os

try:
	os.remove('./results/history_simulation.pickle')
except FileNotFoundError:
	pass

try:
	os.remove('./results/acc.csv')
except FileNotFoundError:
	pass

import argparse


dataset_name = 'MotionSense'
selection_method = 'Random' #Random, POC, All
cluster_metric = 'weights' #CKA, weights
metric_layer = -1 #-1, -2, 1
cluster_method = 'KCenter' #Affinity, HC, KCenter, Random
POC_perc_of_clients = 0.5
n_clients = 12
n_rounds = 15
n_clusters = 4
clustering = True
cluster_round = 5
non_iid = True
Xnon_iid = False

for i in ['Random', 'POC', 'All']:
	for j in ['wights', 'CKA']:
		for k in ['KCenter', 'HC', 'Random']:
			for m in ['-1', '-2']:
				


def funcao_cliente(cid, 
		   n_clients=n_clients,
		    dataset=dataset_name, non_iid=non_iid, model_name = 'DNN',
			local_epochs = 1, n_rounds = n_rounds, n_clusters = n_clusters, Xnon_iid=Xnon_iid,
			selection_method = selection_method, 
			POC_perc_of_clients = POC_perc_of_clients,
			cluster_metric = cluster_metric,
			metric_layer = metric_layer,
			cluster_method = cluster_method):
	return ClientBase(int(cid), n_clients=n_clients,
		    dataset=dataset_name, non_iid=non_iid, model_name = 'DNN',
			local_epochs = 1, n_rounds = n_rounds, n_clusters = n_clusters, Xnon_iid=Xnon_iid,
			selection_method = selection_method, 
			POC_perc_of_clients = POC_perc_of_clients,
			cluster_metric = cluster_metric,
			metric_layer = metric_layer,
			cluster_method = cluster_method)

history = fl.simulation.start_simulation(client_fn=funcao_cliente, 
								num_clients=n_clients, 
								strategy=NeuralMatch(model_name='DNN',  n_clients = n_clients, 
			     									clustering = clustering, clustering_round = cluster_round, 
													n_clusters = n_clusters, dataset=dataset_name, fraction_fit=1, 
													selection_method = selection_method, 
													POC_perc_of_clients = POC_perc_of_clients,
													cluster_metric = cluster_metric,
													metric_layer = metric_layer,
													cluster_method = cluster_method),
								config=fl.server.ServerConfig(n_rounds))