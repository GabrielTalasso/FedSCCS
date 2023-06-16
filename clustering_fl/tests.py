import pandas as pd

n_clients = 7
data_path = 'clustering_fl/data'

data_perc = 0.01 #percentual de dados que serão compartilhados de cada cliente
x_servidor = pd.DataFrame()

for i in range(n_clients):
  with open(f'{data_path}/client{i+1}.csv', 'rb') as train_file:
    data_client_i = pd.read_csv(train_file).drop('Unnamed: 0', axis = 1) 
    x_servidor = pd.concat([data_client_i.sample(int(len(data_client_i)*data_perc)), x_servidor],
                           ignore_index = True)

y_servidor =  x_servidor['label'].values
x_servidor.drop('label', axis=1, inplace=True)
# train.drop('subject', axis=1, inplace=True)
# train.drop('trial', axis=1, inplace=True)
x_servidor =  x_servidor.values

print(x_servidor[1])