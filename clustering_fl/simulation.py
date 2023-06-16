from client import ClientBase
from server import NeuralMatch
import pickle
import flwr as fl
import os

try:
	os.remove('./clustering_fl/data/history_simulation.pickle')
except FileNotFoundError:
	pass

try:
	os.remove('./clustering_fl/acc.csv')
except FileNotFoundError:
	pass


n_clients = 7
n_rounds = 10

def funcao_cliente(cid):
	return ClientBase(int(cid))

history = fl.simulation.start_simulation(client_fn=funcao_cliente, 
								num_clients=n_clients, 
								strategy=NeuralMatch(fraction_fit=1),
								config=fl.server.ServerConfig(n_rounds))



with open('clustering_fl/data/history_simulation.pickle', 'wb') as file:
    pickle.dump(history, file, protocol=pickle.HIGHEST_PROTOCOL)

#0
#Salvamento automatico no git
#calculo do CKA no servidor (ou na simulação)
#usar ruido para o calculo do CKA
#usar conjunto do cliente no teste!

#rodar com qualquer numero de clientes e nao iid - falar com Filipe

#1)
#fazer clustering com base no CKA por aqui

#2)
#treinar os clientes por cluster
#treina algumas rodadas -> clusteriza -> treina os clusters