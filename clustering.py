import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as spc
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.cluster import AffinityPropagation

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

for i in np.linspace(min_link,max_link, 5000):

  le = len(pd.Series(spc.fcluster(linkage, i, 'distance' )).unique())
  if le == k:
    th = i

idx = spc.fcluster(linkage, th, 'distance' )
print(idx)
#dendrogram(linkage, color_threshold=th)
#plt.show()


af = AffinityPropagation(random_state=0).fit(1 / ckas)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_
print(labels)


from sklearn.cluster import OPTICS
clustering = OPTICS(min_samples=2).fit(1/ ckas)
print(clustering.labels_)


from sklearn.decomposition import PCA 
pca = PCA(n_components=2)
pca = pca.fit(ckas)

sns.set_theme(style='white')
sns.scatterplot(x = pca.components_[0], y = pca.components_[1], hue = idx, palette='tab10')
plt.show()

from sklearn.manifold import TSNE
tsne = TSNE(n_components=3, learning_rate='auto', init='random', perplexity=30).fit_transform(ckas)
sns.scatterplot(x = tsne[:,0], y = tsne[:,1], hue = idx, palette='tab10')
sns.set_theme(style='white')
#plt.show()