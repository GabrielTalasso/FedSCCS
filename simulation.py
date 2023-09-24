from client import ClientBase
from server import FedSCCS
import pickle
import flwr as fl
import os
import sys

try:
	os.remove('./results/history_simulation.pickle')
except FileNotFoundError:
	pass

try:
	os.remove('./results/acc.csv')
except FileNotFoundError:
	pass

import argparse

parser = argparse.ArgumentParser(description='Simulator for Clustering in FL ')
parser.add_argument("-a",  "--dataset", 			dest="dataset", help="", metavar="DATASET")
parser.add_argument("-b",  "--nclients", 			dest="nclients", default=10, help="", metavar="NCLIENTS")
parser.add_argument("-c",  "--nrounds", 			dest="nrounds", default=1, help="", metavar="NROUNDS")
parser.add_argument("-d",  "--nclusters", 			dest="nclusters", default=1, help="", metavar="NCLUSTERS")
parser.add_argument("-e",  "--clustering", 			dest="clustering", default=3600, help="", metavar="CLUSTERING")
parser.add_argument("-f",  "--clusterround", 		dest="clusterround", default=2000, help="", metavar="CLUSTEROUND")
parser.add_argument("-g",  "--noniid", 				dest="noniid", default=1, help="", metavar="NONIID")
parser.add_argument("-a1", "--selectionmethod", 	dest="selectionmethod", default=1, help="", metavar="SELECTIONMETHOD")
parser.add_argument("-b1", "--clustermetric", 		dest="clustermetric", default=1, help="", metavar="CLUSTERMETRIC")
parser.add_argument("-c1", "--metriclayer", 		dest="metriclayer", default=1, help="", metavar="METRICLAYER")
parser.add_argument("-d1", "--pocpercofclients", 	dest="pocpercofclients", default=1, help="", metavar="POCPERCOFCLIENTS")
parser.add_argument("-e1", "--clustermethod", 		dest="clustermethod", default=1, help="", metavar="CLUSTERMETHOD")

options = parser.parse_args()

#dataset_name = 'MotionSense'
#selection_method = 'Random' #Random, POC, All
#cluster_metric = 'weights' #CKA, weights
#metric_layer = -1 #-1, -2, 1
#cluster_method = 'KCenter' #Affinity, HC, KCenter, Random
#POC_perc_of_clients = 0.5
#n_clients = 12
#n_rounds = 5
#n_clusters = 5
#clustering = True
#cluster_round = 2
#non_iid = True
Xnon_iid = False

dataset_name = options.dataset
n_clients = int(options.nclients)
n_rounds = 		int(options.nrounds)
n_clusters = 	int(options.nclusters)
clustering = 	options.clustering
cluster_round = int(options.clusterround)
non_iid = 		options.noniid
selection_method 	= options.selectionmethod   #'Random' #Random, POC, All
cluster_metric 		= options.clustermetric   #'weights' #CKA, weights
metric_layer 		= int(options.metriclayer)    #-1 #-1, -2, 1
cluster_method		= options.clustermethod   #'KCenter' #Affinity, HC, KCenter, Random
POC_perc_of_clients = float(options.pocpercofclients)  #0.5

#verificacao de redundancia

#if (n_clusters != 2) and (cluster_method == 'Affinity'): 
	#para o affinity não importa o numero de cluster, entao rodaremos apenas 1 vez (quando n_clusters == 2)
	#sys.exit()

if (cluster_metric == 'CKA') and (cluster_method == 'KCenter'):
	#o KCenter so funciona com os pesos, não é necessário rodar com o CKA
	sys.exit()

if (n_clusters == 1) and (cluster_method != 'HC'):
	#quando o n_clusters = 1, temos o FedAvg e podemos roda-lo só uma vez (nesse caso escolhemos HC)
	sys.exit()

if (cluster_method == 'Random') and (metric_layer != -1):
	sys.exit()

if (cluster_method == 'Random') and (cluster_metric == 'weights'):
	sys.exit()


def funcao_cliente(cid):
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
								strategy=FedSCCS(model_name='DNN',  n_clients = n_clients, 
			     									clustering = clustering, clustering_round = cluster_round, 
													n_clusters = n_clusters, dataset=dataset_name, fraction_fit=1, 
													selection_method = selection_method, 
													POC_perc_of_clients = POC_perc_of_clients,
													cluster_metric = cluster_metric,
													metric_layer = metric_layer,
													cluster_method = cluster_method),
								config=fl.server.ServerConfig(n_rounds))



with open('./results/history_simulation.pickle', 'wb') as file:
    pickle.dump(history, file, protocol=pickle.HIGHEST_PROTOCOL)


##python3 simulation.py --dataset MotionSense --nclients 24 --nrounds 30 --nclusters 5 --clustering True --clusterround 5 --noniid True --selectionmethod All --clustermetric CKA --metriclayer -1 --clustermethod HC --pocpercofclients 0.5
##python3 simulation.py --dataset MotionSense --nclients 24 --nrounds 30 --nclusters 5 --clustering True --clusterround 5 --noniid True --selectionmethod Random --clustermetric weights --metriclayer -1 --clustermethod KCenter --pocpercofclients 0.5
##python3 simulation.py --dataset MotionSense --nclients 24 --nrounds 30 --nclusters 1 --clustering True --clusterround 5 --noniid True --selectionmethod All --clustermetric weights --metriclayer -1 --clustermethod HC --pocpercofclients 0.5
##python3 simulation.py --dataset MotionSense --nclients 24 --nrounds 30 --nclusters 5 --clustering True --clusterround 5 --noniid True --selectionmethod All --clustermetric weights --metriclayer -1 --clustermethod HC --pocpercofclients 0.5
