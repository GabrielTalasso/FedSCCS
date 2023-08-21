import pandas as pd
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

n_clusters = 10
selection = 'POC'
method = 'HC'


paths = [f'local_logs/MotionSense/CKA-(-1)-{method}-{selection}-0.5/evaluate/acc_24clients_{n_clusters}clusters.csv',
         f'local_logs/MotionSense/weights-(-1)-{method}-{selection}-0.5/evaluate/acc_24clients_{n_clusters}clusters.csv',
         f'local_logs/MotionSense/weights-(-1)-HC-{selection}-0.5/evaluate/acc_24clients_1clusters.csv',
         f'local_logs/MotionSense/weights-(-1)-KCenter-{selection}-0.5/evaluate/acc_24clients_{n_clusters}clusters.csv',
         f'local_logs/MotionSense/CKA-(-1)-Random-{selection}-0.5/evaluate/acc_24clients_{n_clusters}clusters.csv']

for p in paths:
    print(p)
    acc =  pd.read_csv(p , names=['rounds', 'client', 'acc', 'loss'], sep=',',  on_bad_lines='skip')
    print(len(acc))

    if p== paths[0]:
        label_clusters = f'{n_clusters}-CKA-{method}-{selection}'
    
    if p== paths[1]:
        label_clusters = f'{n_clusters}-weights-{method}-{selection}'

    if p== paths[2]:
        label_clusters = f'FedAvg-{selection}'

    if p== paths[3]:
        label_clusters = f'{n_clusters}-KCenter-{selection}'

    if p== paths[4]:
        label_clusters = f'{n_clusters}-Random-{selection}'


    sns.lineplot(acc.groupby('rounds').mean(), y = 'acc', x = 'rounds', legend='brief', label=label_clusters)
plt.show()