import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd

from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as spc
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.cluster import AffinityPropagation
from sklearn.cluster import OPTICS

def server_Hclusters(matrix, k, plot_dendrogram , dataset, n_clients, n_clusters,
                    server_round, cluster_round):

    pdist = spc.distance.pdist(matrix)
    linkage = spc.linkage(pdist, method='ward')
    min_link = linkage[0][2]
    max_link = linkage[-1][2]


    th = max_link
    for i in np.linspace(min_link,max_link, 5000):

        le = len(pd.Series(spc.fcluster(linkage, i, 'distance' )).unique())
        if le == k:
            th = i

    idx = spc.fcluster(linkage, th, 'distance' )
    print(idx)

    if plot_dendrogram and (server_round == cluster_round):

        dendrogram(linkage, color_threshold=th)
        plt.savefig(f'results/clusters_{dataset}_{n_clients}clients_{n_clusters}clusters.png')
    

    return idx

def server_Hclusters2(matrix, plot_dendrogram = False):

    pdist = spc.distance.pdist(matrix)
    linkage = spc.linkage(pdist, method='ward')

    max_link = linkage[-1][2]
    t = max_link/3 #como escolher?
    
    idx = spc.fcluster(linkage, t = t, criterion = 'distance')
    print(idx)

    if plot_dendrogram:
        dendrogram(linkage)
        plt.show()

    return idx

def server_AffinityClustering(matrix):

    af = AffinityPropagation(random_state=0).fit( 1 / matrix )
    idx = af.labels_

    return idx

def server_OPTICSClustering(matrix):
    clustering = OPTICS(min_samples=2).fit(1/ matrix)
    idx = clustering.labels_



def cka(X, Y):

    # Implements linear CKA as in Kornblith et al. (2019)
    X = X.copy()
    Y = Y.copy()

    # Center X and Y
    X -= X.mean(axis=0)
    Y -= Y.mean(axis=0)

    # Calculate CKA
    XTX = X.T.dot(X)
    YTY = Y.T.dot(Y)
    YTX = Y.T.dot(X)

    return (YTX ** 2).sum() / (np.sqrt((XTX ** 2).sum() * (YTY ** 2).sum()))

