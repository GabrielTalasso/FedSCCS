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

dataset_name = 'MNIST'
n_clients = 25
n_rounds = 15
n_clusters = 1
clustering = True
cluster_round = 5
non_iid = False

def funcao_cliente(cid):
	return ClientBase(int(cid), n_clients=n_clients,
		    dataset=dataset_name, non_iid=non_iid, model_name = 'DNN',
			local_epochs = 1, n_rounds = n_rounds, n_clusters = n_clusters)

history = fl.simulation.start_simulation(client_fn=funcao_cliente, 
								num_clients=n_clients, 
								strategy=NeuralMatch(model_name='DNN',  n_clients = n_clients, 
			     									clustering = clustering, clustering_round = cluster_round, 
													n_clusters = n_clusters, dataset=dataset_name),
								config=fl.server.ServerConfig(n_rounds))



with open('./results/history_simulation.pickle', 'wb') as file:
    pickle.dump(history, file, protocol=pickle.HIGHEST_PROTOCOL)



#1)
#fazer clustering com base no CKA por aqui

#- passar a parte do calculo do clustering para o servidor
#- como treinar varios modelos no servidor?


#treina algumas rodadas -> clusteriza -> treina os clusters