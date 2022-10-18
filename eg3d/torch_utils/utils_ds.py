import numpy as np
import torch
import random
import torch.nn as nn

from ipdb import set_trace as st

def grp_range_torch(a,dev):
    idx = torch.cumsum(a,0)
    # st()
    id_arr = torch.ones(idx[-1],dtype = torch.int64,device=dev)
    id_arr[0] = 0
    id_arr[idx[:-1]] = -a[:-1]+1
    return torch.cumsum(id_arr,0)
    # generate array like [0,1,2,3,4,5,0,1,2,3,4,5,6] where each 0-n gives id to points inside the same grid

def parallel_FPS(np_cat_fea,K):
    return  nb_greedy_FPS(np_cat_fea,K)

def nb_greedy_FPS(xyz,K):
    start_element = 0
    sample_num = xyz.shape[0]
    sum_vec = np.zeros((sample_num,1),dtype = np.float32)
    xyz_sq = xyz**2
    for j in range(sample_num):
        sum_vec[j,0] = np.sum(xyz_sq[j,:])
    pairwise_distance = sum_vec + np.transpose(sum_vec) - 2*np.dot(xyz, np.transpose(xyz))

    candidates_ind = np.zeros((sample_num,),dtype = np.bool_)
    candidates_ind[start_element] = True
    remain_ind = np.ones((sample_num,),dtype = np.bool_)
    remain_ind[start_element] = False
    all_ind = np.arange(sample_num)

    for i in range(1,K):
        if i == 1:
            min_remain_pt_dis = pairwise_distance[:,start_element]
            min_remain_pt_dis = min_remain_pt_dis[remain_ind]
        else:
            cur_dis = pairwise_distance[remain_ind,:]
            cur_dis = cur_dis[:,candidates_ind]
            min_remain_pt_dis = np.zeros((cur_dis.shape[0],),dtype = np.float32)
            for j in range(cur_dis.shape[0]):
                min_remain_pt_dis[j] = np.min(cur_dis[j,:])
        next_ind_in_remain = np.argmax(min_remain_pt_dis)
        next_ind = all_ind[remain_ind][next_ind_in_remain]
        candidates_ind[next_ind] = True
        remain_ind[next_ind] = False

    return candidates_ind