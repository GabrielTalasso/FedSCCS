import pandas as pd
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = 'MNIST'
n_rounds = 15
n_clients = 10
n_clusters = [1,2, 5,8, 'Affinity']
#n_clusters = [ 1, 5, 10, 15, 'Affinity']

path = './experiments/MNIST/noniid'
#path = './experiments/MNIST/noniid'

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

## comparing with random clusters:
c = 5
for i in range(1,4):
    acc_random = pd.read_csv(f'{path}/random_clusters/acc{i}_{dataset}_{n_clients}clients_{c}clusters.csv',
                       names=['_', 'client', 'acc', 'loss']).drop('_', axis = 1)
    acc_random['round'] = rounds

    sns.lineplot(acc_random.groupby('round').mean(), y = 'acc', x = 'round', legend='brief', label=f'random{i}')

acc =  pd.read_csv(f'{path}/acc_{dataset}_{n_clients}clients_{c}clusters.csv',
                       names=['_', 'client', 'acc', 'loss']).drop('_', axis = 1)
acc['round'] = rounds
sns.lineplot(acc.groupby('round').mean(), y = 'acc', x = 'round', legend='brief', label='cka_clusters')

plt.show()

