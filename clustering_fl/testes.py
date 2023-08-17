{C_1, C_2, ... , C_K} = clustering({c_1, c_2, ... , c_i})

for t in range(T):

    for k in range(K):

        for i in C_k:
            w^i(t+1) = LocalUpdate(C_i)

        w_k(t+1) = FedAvg(C_k)
