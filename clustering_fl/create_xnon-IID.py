from fedartml import InteractivePlots
from keras.datasets import cifar10

# Load CIFAR 10data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Define (centralized) labels to use 
CIFAR10_labels = y_train
# Instanciate InteractivePlots object

my_plot = InteractivePlots(labels = CIFAR10_labels)

# Show plot
my_plot.show_stacked_distr_percent_noniid()

from fedartml import SplitAsFederatedData
from keras.datasets import cifar10
import numpy as np

# Define random state for reproducibility
random_state = 0

# Load data
(x_train_glob, y_train_glob), (x_test_glob, y_test_glob) = cifar10.load_data()
y_train_glob = np.reshape(y_train_glob, (y_train_glob.shape[0],))
y_test_glob = np.reshape(y_test_glob, (y_test_glob.shape[0],))

# Normalize pixel values to be between 0 and 1
x_train_glob, x_test_glob = x_train_glob / 255.0, x_test_glob / 255.0

# Instantiate a SplitAsFedseratedData object
my_federater = SplitAsFederatedData(random_state = random_state)
# Get federated dataset from centralized dataset
_, list_ids_sampled_dic, miss_classes, dists = my_federater.create_clients(image_list = x_train_glob, label_list = y_train_glob, 
                                                             num_clients = 2, method = "percent_noniid", percent_noniid = 50)

print(list_ids_sampled_dic)