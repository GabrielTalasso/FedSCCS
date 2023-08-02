l = [1,2,3,4,68000]
l.remove(68000)
print(l)
import numpy as np

#print(np.where(a < 60, a, ))

a= '12345'
print(int(a[:-1]))

import torch
idx_train = torch.load(f'data/CIFAR10/Cifar10_train/sim_motiv_50_cli_exp_alpha_0.3_id_{1}.pt')
print(len(idx_train[10]))
