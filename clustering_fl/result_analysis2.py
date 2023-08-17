import pandas as pd
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = 'MotionSense'

paths = ['local_logs/MotionSense/CKA-(-1)-HC-All-0.5/evaluate/acc_24clients_5clusters.csv',
         'local_logs/MotionSense/weights-(-1)-KCenter-Random-0.5/evaluate/acc_24clients_5clusters.csv',
         'local_logs/MotionSense/weights-(-1)-HC-All-0.5/evaluate/acc_24clients_1clusters.csv',
         'local_logs/MotionSense/weights-(-1)-HC-All-0.5/evaluate/acc_24clients_5clusters.csv',
         'local_logs/MotionSense/CKA-(-1)-HC-Random-0.5/evaluate/acc_24clients_5clusters.csv',
         'local_logs/MotionSense/weights-(-1)-KCenter-All-0.5/evaluate/acc_24clients_5clusters.csv']
for p in paths:
    acc =  pd.read_csv(p , names=['rounds', 'client', 'acc', 'loss'])

    if p== paths[0]:
        label_clusters = '5HC-CKA-All'
    
    if p== paths[1]:
        label_clusters = 'KCenter-Random'

    if p== paths[2]:
        label_clusters = 'FedAvg-All'

    if p== paths[3]:
        label_clusters = '5HC-weights-All'

    if p== paths[4]:
        label_clusters = '5HC-CKA-Random'

    if p== paths[5]:
        label_clusters = 'KCenter-All'


    sns.lineplot(acc.groupby('rounds').mean(), y = 'acc', x = 'rounds', legend='brief', label=label_clusters)
plt.show()