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

dataset_name = 'MotionSense'
selection_method = 'POC' #Random, POC, All
cluster_metric = 'CKA' #CKA, weights
metric_layer = -1 #-1, -2, 1
cluster_method = 'HC' #Affinity, HC, KCenter, Random
POC_perc_of_clients = 0.5
n_clients = 12
n_rounds = 5
n_clusters = 5
clustering = True
cluster_round = 2
non_iid = True
Xnon_iid = False

for i in ['POC', 'Random', 'All']:
	for j in ['CKA', 'weights']:
		for k in ['HC', 'Affinity', 'HC', 'KCenter', 'Random']:
			for n in [1, 3, 5]:
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
                                                strategy=NeuralMatch(model_name='DNN',  n_clients = n_clients, 
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


##python3 simulation.py --dataset MNIST --nclients 10 --nrounds 5 --nclusters 5 --clustering True --clusterround 2 --noniid True 


#escrever comandos sh
#colocar novos params no init: sim_metric, clustering method, selection metrics
#criar novas pastas para salvar resultados