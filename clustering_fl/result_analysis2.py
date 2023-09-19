import pandas as pd
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


n_clusters = 10
selection = 'All'
method = 'HC'


paths = ['local_logs/MotionSense/CKA-(-1)-HC-All-0.5/evaluate/acc_24clients_10clusters.csv',
         'local_logs/MotionSense/CKA-(-1)-Random-All-0.5/evaluate/acc_24clients_10clusters.csv',
         'acc_24clients_10clusters.csv']

for p in paths:
    print(p)
    acc =  pd.read_csv(p , names=['rounds', 'client', 'acc', 'loss'], sep=',',  on_bad_lines='skip')
    print(len(acc))

    if p== paths[0]:
        label_clusters = f'{n_clusters}-CKA-{method}-{selection}wo/ Data'
    if p== paths[1]:
        label_clusters = f'{n_clusters}-CKA-Random-{selection}'
    if p== paths[2]:
        label_clusters = f'{n_clusters}-CKA-Random-{selection}-w/ Data'


    sns.lineplot(acc.groupby('rounds').mean(), y = 'acc', x = 'rounds', legend='brief', label=label_clusters)
plt.show()


