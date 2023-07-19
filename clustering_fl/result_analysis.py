import pandas as pd
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = 'MNIST'
n_rounds = 15
n_clients = 25
n_clusters = [1,5,8]
n_clusters = [ 1, 5]#, 10,15]

path = './experiments/iid'
path = './results'

for c in n_clusters:
    acc =  pd.read_csv(f'{path}/acc_{dataset}_{n_clients}clients_{c}clusters.csv',
                       names=['_', 'client', 'acc', 'loss']).drop('_', axis = 1)
    if c==1:
        label_clusters = 'FedAvg'
    else:
        label_clusters = f'{c} clusters'

    rounds = np.ones(n_clients)
    for r in range(2,n_rounds+1):
        rounds = np.concatenate((rounds, np.ones(n_clients)*r), axis = None)
    acc['round'] = rounds

    sns.lineplot(acc.groupby('round').mean(), y = 'acc', x = 'round', legend='brief', label=label_clusters)
plt.show()



#acc =  pd.read_csv('./results/acc.csv', names=['_', 'client', 'acc', 'loss']).drop('_', axis = 1)
#
#n_clients = acc['client'].max() + 1
#n_rounds = int(len(acc) / n_clients)
#
#acc_fedavg = pd.read_csv(f'./experiments/acc_fedavg_{n_clients}c.csv', names=['_', 'client', 'acc', 'loss']).drop('_', axis = 1)
#
#print(n_rounds)
#
#rounds = np.ones(n_clients)
#for r in range(2,n_rounds+1):
#    rounds = np.concatenate((rounds, np.ones(n_clients)*r), axis = None)
#
#acc['round'] = rounds
#acc_fedavg['round'] = rounds
#
#
#sns.lineplot(data = acc, y = 'acc', x = 'round',  hue =  'client')
#plt.show()
#
#sns.lineplot(acc.groupby('round').mean(), y = 'acc', x = 'round', legend='brief', label='with_clustering')
#sns.lineplot(acc_fedavg.groupby('round').mean(), y = 'acc', x = 'round', legend='brief', label='fedavg')
#plt.show()
#
#sns.histplot(acc[acc['round'] == n_rounds], x = 'acc')
#plt.show()