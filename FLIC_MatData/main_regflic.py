# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/main_fed.py
# credit goes to: Paul Pu Liang

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# Adapted from the code of Collins et al.
#
# Author A.R
#%%
import argparse
import os
import random
import copy
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt



import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple, Dict
from argparse import Namespace

from Het_Update import Het_LocalUpdate, het_test_img_local_all, train_preproc
from Het_Nets import get_reg_model, get_preproc_model
from sklearn.model_selection import train_test_split


import time
import warnings
warnings.filterwarnings("ignore")

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cpu")


def dict_to_namespace(d: dict):
    namespace = Namespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(namespace, k, dict_to_namespace(v))
        else:
            setattr(namespace, k, v)
    return namespace


def standard_process(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, np.ndarray]:
    # df = df.dropna(axis=0, how='any')
    target = df['e_form'].values

    cols_to_drop = [
        "formula", "atom a", "atom b", "lowest distortion",
        "e_form"
    ]
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]  # 仅保留存在的列
    df.drop(cols_to_drop, axis=1, inplace=True)  # 删除存在的列，如果不存在则忽略错误
    print("列删除后剩余的列名:", df.columns)  # 打印列名以检查
    #x = df.drop(columns=['e_form'], axis=1, errors='ignore')  # 删除存在的列，如果不存在则忽略错误
    ss = StandardScaler()
    pt = ss.fit_transform(df)

    df = pd.DataFrame(pt,columns=df.columns)

    
    return df, target

def prepare_data(
    args
) -> Tuple[Dict[int, DataLoader], Dict[int, int]]:
    '''
    prepare data for training and testing
    '''
    train_data_paths = args.train_data_paths
    num_users = args.num_users
    batch_size = args.batch_size
    
     # 确保数据集数量与用户数量匹配
    assert len(train_data_paths) == num_users, "数据集数量必须与用户数量相匹配"
    def create_data_loader(df: pd.DataFrame, batch_size: int) -> DataLoader:

        x, y = standard_process(df)
        feature_number = x.shape[1]
        print(feature_number)
        x_tensor = torch.tensor(x.values, dtype=torch.float32, device=device)

        y_tensor = torch.tensor(y, dtype=torch.float32, device=device).view(-1, 1)

        dataset = TensorDataset(x_tensor, y_tensor)

        return DataLoader(dataset=dataset, batch_size=batch_size, drop_last = True, shuffle=True), feature_number
    
    
    user_data = {}
    f_num = {}
    #user_test_data = {}
    #通过循环为每个用户创建了一个 DataLoader，并将其存储在 user_data 字典中
    for i in range(args.num_users):
        df_train = pd.read_csv(train_data_paths[i])
        # print(df_train)
        dataset_train , feature_number = create_data_loader(df_train, batch_size)
        user_data[i] = dataset_train
        f_num[i] = feature_number
        #print(df_train) 
    #print(user_data_addeddim)
    #通过循环为每个用户创建了一个 DataLoader，并将其存储在 user_data 字典中
    return user_data, f_num


if __name__ == '__main__':
    import sys
    #sys.argv = ['']
    parser = argparse.ArgumentParser()

    parser.add_argument('--alg', type=str, default='FLic', help="Algorithm")
    #parser.add_argument('--dataset', type=str, default='ABX3', help="choice of the dataset")

    parser.add_argument('--num_users', type=int, default=3, help="number of users")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument('--shard_per_user', type=int, default=2, help="number of classes per user")
    #parser.add_argument('--num_classes', type=int, default=10, help="number of classes")

    parser.add_argument('--frac', type=float, default=1, help="the fraction of clients: C")

    parser.add_argument('--epochs', type=int, default=100,help="rounds of training")
    parser.add_argument('--local_bs', type=int, default=100, help="local batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate for model")
    parser.add_argument('--local_ep', type=int, default=10, help="number of local epoch")
    parser.add_argument('--local_rep_ep', type=int, default=1, help="number of local epoch for representation among local_ep")
    parser.add_argument('--reg_w', type=float, default=0.001, help="regularization of W ")
    parser.add_argument('--reg_reg_prior', type=float, default=0.001, help="regularization of W ")


    parser.add_argument('--model_type', type=str, default='reg', help="choosing the global model, [classif, no-hlayers, 2-hlayers]")
    parser.add_argument('--n_hidden', type=int, default=64, help="number of units in hidden layers")
    parser.add_argument('--dim_latent', type=int, default=16, help="latent dimension")
    parser.add_argument('--align_epochs', type=int, default=50, help="number of epochs for alignment during pretraining")
    parser.add_argument('--align_epochs_altern', type=int, default=3, help="number of epochs for alignment during alternate")
    parser.add_argument('--align_lr', type=float, default=0.001, help="learning rate of alignment ")
    parser.add_argument('--align_bs', type=int, default=10, help="batch_size for alignment")
    parser.add_argument('--distance', type=str, default='wd', help="distance for alignment")

    parser.add_argument('--mean_target_variance', type=int, default=10, help="std of random prior means")
    parser.add_argument('--update_prior', type=int, default=1, help= "updating prior (1 for True)")
    parser.add_argument('--update_net_preproc', type=int, default=1, help= "updating preproc network")
    parser.add_argument('--start_optimize_rep', type=int, default=20, help= "starting iterations for global model optim")

    parser.add_argument('--seed', type=int, default=0,help="choice of seed")
    parser.add_argument('--gpu', type=int, default=-1, help="gpu to use (if any")
    parser.add_argument('--num_workers', type=int, default=0, help="number of workers in dataloader")
    parser.add_argument('--test_freq', type=int, default=1, help="frequency of test eval")
    parser.add_argument('--timing', type=int, default=0, help="compute running time")
    parser.add_argument('--savedir', type=str, default='./save/', help="save dire")
    
    parser.add_argument("--train_data_paths", type=str, nargs='+', default=[
        f"{PROJECT_PATH}/data/train_part0.csv",
        f"{PROJECT_PATH}/data/train_part1.csv",
        f"{PROJECT_PATH}/data/train_part2.csv"
    ])
    #parser.add_argument("--test_data_path", type=str, default=f"{PROJECT_PATH}/data/test_dataset.csv")
    

    args = parser.parse_args()
    
    
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    
   # lens = np.ones(args.num_users)

    user_data ,f_num = prepare_data(args)#11111

        
    lens = np.ones(args.num_users)#1111
    
       
    print(args)
    if args.model_type =='average':
        # for FedAverage, use one local epoch and use it for the global part
        # and the appropriate model
        args.local_ep = 1
        args.local_rep_ep = 1    
    
    # - select global model and trigger the train mode
    #   set appropriate arguments to make it trainable
    # - show model
    
    net_glob, w_glob_keys = get_reg_model(args)#11111
    net_glob.train()#11111
    
    total_num_layers = len(net_glob.state_dict().keys())#1111
    print(total_num_layers)
    print(net_glob.state_dict().keys())
    
    w_locals = {}#11111
    for user in range(args.num_users):#1111111
        w_local_dict = {}
        for key in net_glob.state_dict().keys():
            w_local_dict[key] = net_glob.state_dict()[key]
        w_locals[user] = w_local_dict
        
        
    from prior_reg import Prior
    prior = Prior([args.dim_latent],args.mean_target_variance)#11111

    
    net_preprocs = {}#111111
    for user in range(args.num_users):#11111111
        # 确保 f_num[user] 是整数，并且是每个用户数据集的特征数目
        if not isinstance(f_num[user], int) or f_num[user] <= 0:
            raise ValueError(f"The feature number for user {user} is not a valid integer")
        # 使用 f_num[user] 作为 dim_in 参数创建模型
        net_preproc = get_preproc_model(args, dim_in=f_num[user], dim_out=args.dim_latent)

        net_preprocs[user]=(net_preproc)
    
    # #%%
    print(net_preprocs[0])

    train_preproc(net_preprocs, user_data, prior,
                  n_epochs=args.align_epochs,
                  args=args,
                  verbose=True)#111111

    
    indd = None   
    loss_train = []
    loss_local_full = [[] for _ in range(args.num_users)]


    times = []
    start = time.time()


    
    for iter in range(args.epochs+1):

        w_glob = {}
        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)
        if iter == args.epochs:
            m = args.num_users
            
        #idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        idxs_users = np.arange(args.num_users)
        w_keys_epoch = w_glob_keys
        times_in = []
        total_len=0
        
        
        for ind, idx in enumerate(idxs_users):

            print(ind,'user:',idx)
            start_in = time.time()
            
            # instantiate an algo for localupdate
            local = Het_LocalUpdate(args=args, dataset= user_data[idx],current_iter=iter)
            local.update_prior = args.update_prior 
            #-----------------------------------------------------------------
            # starting local update
            # 1. initialize a curent local model (the one of the user to be update) with the global model 
            # 2. replace weights that are local with current status of local model for that users 
            # 3. update local model net_local -- 
            #-----------------------------------------------------------------
            net_local = copy.deepcopy(net_glob)
            w_local = net_local.state_dict()
            for k in w_locals[idx].keys():
                if k not in w_glob_keys:
                    w_local[k] = w_locals[idx][k]
            net_local.load_state_dict(w_local)
            #-----------------------------------------------------------------
            # updating local models
            #------------------------------------------------------------------
            

            last = iter == args.epochs
            
            w_local, loss, indd, last_loss = local.train(net=net_local.to(args.device), 
                                              net_preproc=net_preprocs[idx].to(args.device),
                                              w_glob_keys=w_glob_keys, 
                                              prior=prior,
                                              lr=args.lr, last=last)
            loss_locals.append(copy.deepcopy(loss))
            total_len += lens[idx]
            loss_local_full[idx] = loss_local_full[idx] + copy.deepcopy(last_loss)
            
        
            # lens are the weigth of local model when doing averages
            # summing all the weights of shared global models
            if len(w_glob) == 0:
                # first iteration 
                w_glob = copy.deepcopy(w_local)
                for k,key in enumerate(net_glob.state_dict().keys()):
                    #key_full = '1.' + key
                    w_glob[key] = w_glob[key]*lens[idx]
                    w_locals[idx][key] = w_local[key]
            else:
                for k,key in enumerate(net_glob.state_dict().keys()):

                    if key in w_glob_keys:
                        w_glob[key] += w_local[key]*lens[idx]

                    w_locals[idx][key] = copy.deepcopy(w_local[key])

            times_in.append( time.time() - start_in )
            #print(times_in[-1])

        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)
        #--------------------------------------------------------------------
        # get weighted average for global weights
        # by normalized with respect to total_len
        #----------------------------------------------------------------------
        for k in net_glob.state_dict().keys():
            w_glob[k] = torch.div(w_glob[k], total_len)

        #----------------------------------------------------------------------
        # updating global model 
        # and initializing current local model with global
        #----------------------------------------------------------------------
        w_local = net_glob.state_dict()
        for k in w_glob_keys:
            w_local[k] = w_glob[k]
        if args.epochs != iter:
            net_glob.load_state_dict(w_glob)
        
        
        #----------------------------------------------------------------------
        # updating prior model 
        #----------------------------------------------------------------------
        prior.mu = prior.mu_temp/ prior.n_update
        prior.init_mu_temp()        

        if args.align_epochs_altern >0:
   
            #----------------------------------------------------------------------
            #  Reajust mapping with respect to the new means
            #----------------------------------------------------------------------
            prior.mu.requires_grad_(False)
            train_preproc(net_preprocs,user_data,prior,
                          n_epochs=args.align_epochs_altern,
                          args=args,
                          verbose=True
                          )
        #
        #    verbosity
        #
   
        if iter % args.test_freq==args.test_freq-1 or iter>=args.epochs-10:
            if times == []:
                times.append(max(times_in))
            else:
                times.append(times[-1] + max(times_in))
                
            # avg_test_mse, avg_test_rmse, test_loss, avg_test_mae, avg_test_r2 = het_test_img_local_all(net_glob, net_preprocs, args,
            #                                              user_test_data,
            #                                             w_glob_keys=w_glob_keys,
            #                                             w_locals=w_locals,indd=indd)
            avg_train_mse, avg_train_rmse, train_loss, avg_train_mae, avg_train_r2 = het_test_img_local_all(net_glob, net_preprocs, args,
                                                         user_data,
                                                        w_glob_keys=w_glob_keys,
                                                        w_locals=w_locals,indd=indd)

            # for algs which learn a single global model, these are the local accuracies (computed using the locally updated versions of the global model at the end of each round)
            print('Round {:3d}, Train loss: {:.3f}, Train MSE: {:.3f}, Train RMSE: {:.3f}, Train MAE: {:.3f}, Train R2-score: {:.2f}'.format(
                        iter, train_loss, avg_train_mse, avg_train_rmse, avg_train_mae, avg_train_r2))
            regression_metrics = pd.DataFrame({
                'MSE': [avg_train_mse],
                'RMSE': [avg_train_rmse],
                'MAE': [avg_train_mae],
                'R2 Score': [avg_train_r2]
            })

    end = time.time()
    print(end-start)
    print(times)
    #print(accs)
    print(args)
    
    
    #%%
    save_dir = args.savedir
    out_dir = f"{args.num_users:d}-{args.shard_per_user:d}-frac{args.frac:2.1f}"
    out_dir += f"-upd_p{args.update_prior:}-upd_prp{args.update_net_preproc:}"
    out_dir += f"-reg_w{args.reg_w:2.3f}--reg_w{args.reg_reg_prior:2.3f}/"
    
    opt = copy.deepcopy(vars(args))    
    for keys in ['full_client_freq','num_workers','device','mean_target_variance',
                  'align_epochs_altern','num_classes','subsample_client_data',
                  'gpu','test_freq','n_per_class','dataset','num_users','shard_per_user','frac',
                  'update_prior','update_net_preproc','reg_w','reg_reg_prior','savedir', 'train_data_paths', 'test_data_path']:
        opt.pop(keys, None)
    
    key_orig = ['local_ep','local_rep_ep','model_type']
    key_new = ['l_ep','l_repep','mdl','reg_cp']
    
    if not (os.path.isdir(save_dir+out_dir)):
        os.makedirs(save_dir+out_dir)
        
    
    filename = ""
    for key in opt.keys():
        val = str(opt[key])
        if key_orig.count(key)>0:
            filename += f"{key_new[key_orig.index(key)]}-{val}-"
        else:
            filename += f"{key}-{val}-" 

    base_dir = filename + '.csv'
    base_dir_lc = filename + 'Loss_Curve.csv'
    user_save_path = base_dir

    times= np.array(times)
    
    regression_metrics.to_csv(save_dir+out_dir+base_dir,index=False)
    
    loss_curve = pd.DataFrame(loss_local_full)
    print(loss_curve.index)
    loss_curve.to_csv(save_dir + base_dir_lc, index=False)
    print(loss_curve.columns)
    
    plt.figure(figsize=(10, 6))

    for i in range(args.num_users):
        plt.plot(loss_local_full[i],label=f'User {i}')
        
    
    plt.legend()
    plt.title('Loss Curve per User')
    plt.xlabel('Training Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

        