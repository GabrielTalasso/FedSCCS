import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as spc
from scipy.cluster.hierarchy import dendrogram, linkage

sns.set()

file_path = './data/ckas.pickle'

with open(file_path, "rb") as f:
  ckas = pickle.load(f)

pdist = spc.distance.pdist(ckas)
linkage = spc.linkage(pdist, method='ward')
#idx = spc.fcluster(linkage, 1.8, 'distance' )
#print(idx)

#dendrogram(linkage, color_threshold=1.8)
#plt.show()

min_link = linkage[0][2]
max_link = linkage[-1][2]

k = 3

for i in np.linspace(min_link,max_link, 100):

  le = len(pd.Series(spc.fcluster(linkage, i, 'distance' )).unique())
  if le == k:
    th = i

idx = spc.fcluster(linkage, th, 'distance' )
print(idx)
dendrogram(linkage, color_threshold=th)
plt.show()
