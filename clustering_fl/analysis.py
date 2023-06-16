import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


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

#abre a simulação ja executada
with open('clustering_fl/data/history_simulation.pickle', 'rb') as file:
    history = pickle.load(file)

#salva os id dos clientes, na ordem que foram treinados
clientes = history.metrics_distributed['cids'][0][1]

#ativações das redes
actvs = history.metrics_distributed['actv_last'][0][1]
 #ativações de todos os clientes na ultima camada no ultimo round

matrix = np.zeros((len(actvs), len(actvs)))

#comparando todos com todos
for i , a in enumerate(actvs):
  for j, b in enumerate(actvs):

    x = int(clientes[i])
    y = int(clientes[j])

    matrix[x][y] = cka(a, b)

#salvando as ativações
for i in range(7):
  with open(f'clustering_fl/data/acvt_{i}', "wb") as f:
      pickle.dump(actvs[i], f)

nome_arquivo = "clustering_fl/data/ckas.pickle"
# Salva a matriz no arquivo pickle
with open(nome_arquivo, "wb") as f:
    pickle.dump(matrix, f)


sns.heatmap(matrix, vmin = 0, vmax = 1)
plt.show()