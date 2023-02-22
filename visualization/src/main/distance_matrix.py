import numpy as np
import sys
import os
import persim

nodelist = np.load("/home/zluo/new_chip_design/nodelist.npy")
#nodelist.remove(35)
w_d_lst = []
for idx1 in range(len(nodelist)):
    s_n = nodelist[idx1]
    #dgm1 = np.load(f"/home/zluo/new_chip_design/dgms/pd_back_{s_n}.npy", allow_pickle=True)[-1]
    dgm1 = np.load(f"/home/zluo/new_chip_design/dgms/pd_{s_n}.npy", allow_pickle=True)[-1]
    dgm1 = np.array([tp[-1] for tp in dgm1])
    inner_lst = []
    for idx2 in range(idx1, len(nodelist)):
        #dgm2 = np.load(f"/home/zluo/new_chip_design/dgms/pd_back_{nodelist[idx2]}.npy", allow_pickle=True)[-1]
        dgm2 = np.load(f"/home/zluo/new_chip_design/dgms/pd_{nodelist[idx2]}.npy", allow_pickle=True)[-1]
        dgm2 = np.array([tp[-1] for tp in dgm2])
        inner_lst.append(persim.sliced_wasserstein(dgm1, dgm2, M=50))
    
    w_d_lst.append(inner_lst)
    if idx1//500 > 0 and idx1%500 == 0:
        print(idx1)
        np.save("/home/zluo/new_chip_design/w_d_lst_1.npy", w_d_lst)
        
np.save("/home/zluo/new_chip_design/w_d_lst_1.npy", w_d_lst)
