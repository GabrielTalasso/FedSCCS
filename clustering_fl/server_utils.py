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

class GreedyKCenter(object):
    def fit(self, points, k):
        centers = []
        centers_index = []
        # Initialize distances
        distances = [np.inf for u in points]
        # Initialize cluster labels
        labels = [np.inf for u in points]

        for cluster in range(k):
            # Let u be the point of P such that d[u] is maximum
            u_index = distances.index(max(distances))
            u = points[u_index]
            # u is the next cluster center
            centers.append(u)
            centers_index.append(u_index)

            # Update distance to nearest center
            for i, v in enumerate(points):
                distance_to_u = self.distance(u, v)  # Calculate from v to u
                if distance_to_u < distances[i]:
                    distances[i] = distance_to_u
                    labels[i] = cluster

            # Update the bottleneck distance
            max_distance = max(distances)

        # Return centers, labels, max delta, labels
        self.centers = centers
        self.centers_index = centers_index
        self.max_distance = max_distance
        self.labels = labels

    @staticmethod
    def distance(u, v):
        displacement = u - v
        return np.sqrt(displacement.dot(displacement))

def server_Hclusters(matrix, k, plot_dendrogram , dataset, n_clients, n_clusters,
                    server_round, cluster_round, path):

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
        #plt.savefig(f'results/clusters_{dataset}_{n_clients}clients_{n_clusters}clusters.png')
        plt.savefig(path+f'clusters_{n_clients}clients_{n_clusters}clusters.png')

    

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

def server_KCenterClustering(weights, k):

    KCenter = GreedyKCenter()
    KCenter.fit(weights, k)
    idx = KCenter.labels

    return idx



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

