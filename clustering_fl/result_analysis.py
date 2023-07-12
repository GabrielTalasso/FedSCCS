import pandas as pd
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

acc =  pd.read_csv('./results/acc.csv', names=['_', 'client', 'acc', 'loss']).drop('_', axis = 1)
acc_fedavg = pd.read_csv('./results/acc_fedavg.csv', names=['_', 'client', 'acc', 'loss']).drop('_', axis = 1)

n_clients = acc['client'].max() + 1
n_rounds = int(len(acc) / n_clients)

print(n_rounds)

rounds = np.ones(n_clients)
for r in range(2,n_rounds+1):
    rounds = np.concatenate((rounds, np.ones(n_clients)*r), axis = None)

acc['round'] = rounds
acc_fedavg['round'] = rounds


sns.lineplot(data = acc, y = 'acc', x = 'round',  hue =  'client')
plt.show()

sns.lineplot(acc.groupby('round').mean(), y = 'acc', x = 'round', legend='brief', label='with_clustering')

sns.lineplot(acc_fedavg.groupby('round').mean(), y = 'acc', x = 'round', legend='brief', label='fedavg')

plt.show()

sns.histplot(acc[acc['round'] == n_rounds], x = 'acc')
plt.show()