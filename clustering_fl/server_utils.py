import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd

from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as spc
from scipy.cluster.hierarchy import dendrogram, linkage

def server_Hclusters(matrix, k = 3, plot_dendrogram = False):

    pdist = spc.distance.pdist(matrix)
    linkage = spc.linkage(pdist, method='ward')
    min_link = linkage[0][2]
    max_link = linkage[-1][2]


    th = max_link
    for i in np.linspace(min_link,max_link, 100):

        le = len(pd.Series(spc.fcluster(linkage, i, 'distance' )).unique())
        if le == k:
            th = i

    idx = spc.fcluster(linkage, th, 'distance' )
    print(idx)

    if plot_dendrogram:

        dendrogram(linkage, color_threshold=th)
        plt.show()

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

    return (YTX ** 2).sum() / np.sqrt((XTX ** 2).sum() * (YTY ** 2).sum())

