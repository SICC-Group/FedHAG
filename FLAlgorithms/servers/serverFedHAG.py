from FLAlgorithms.users.userFedHAG import UserFedHAG
#from FLAlgorithms.users.userpFedGen import RegressionTracker
from FLAlgorithms.servers.serverbase import Server
from FLAlgorithms.trainmodel.RKLDivLoss import RKLDivLoss
from FLAlgorithms.users.userpFedGen import RegressionTracker
from utils.model_utils import create_model,convert_data
from utils.model_utils import read_data, read_user_data, aggregate_user_data, create_generative_model, create_discriminator_model
from Het_Update import Het_LocalUpdate, het_test_img_local_all, train_preproc,aggregate_models
from Het_Nets import get_reg_model, get_preproc_model
from sklearn.decomposition import PCA
from FLAlgorithms.users.userbase import User
#from FLAlgorithms.users.userbase import clone_model_paramenter
from torch.utils.data import TensorDataset, DataLoader
from prior_reg import Prior
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
import os
import copy
import time
import itertools
MIN_SAMPLES_PER_LABEL=1

class FedHAG(Server):
    def __init__(self, args, model, seed):
        super().__init__(args, model, seed)

        # Initialize data for all users
        self.data = read_data(args.dataset)
        # data contains: clients, groups, train_data, test_data, proxy_data
        clients = self.data[0]
        total_users = len(clients)
        self.total_test_samples = 0
        self.local = 'local' in self.algorithm.lower() #false
        self.use_adam = 'adam' in self.algorithm.lower() #false
        self.global_mean = None
        self.global_variance=None
        self.early_stop = 20  # stop using generated samples after 20 local epochs
        self.latent_model=create_model(32)
        #self.student_model = copy.deepcopy(self.latent_model) #cnn模型
        self.student_model = copy.deepcopy(self.model) #cnn模型
        self.generator_model_glob = create_generative_model(args.dataset, args.algorithm, args.embedding)#生成器模型
        self.discriminator_model_glob = create_discriminator_model(args.dataset, args.algorithm, args.embedding) # 判别器模型
        if not args.train:
            print('number of generator parameteres: [{}]'.format(self.generator_model_glob.get_number_of_parameters()))
            print('number of discriminator parameteres: [{}]'.format(self.discriminator_model_glob.get_number_of_parameters()))
            print('number of model parameteres: [{}]'.format(self.model.get_number_of_parameters()))
        self.latent_layer_idx = self.generator_model_glob.latent_layer_idx
        self.init_ensemble_configs()
        print("latent_layer_idx: {}".format(self.latent_layer_idx))
        #print("label embedding {}".format(self.generative_model.embedding))
        print("ensemeble learning rate: {}".format(self.ensemble_lr))
        print("ensemeble alpha = {}, beta = {}, eta = {}".format(self.ensemble_alpha, self.ensemble_beta, self.ensemble_eta))
        print("generator alpha = {}, beta = {}".format(self.generative_alpha, self.generative_beta))
        self.init_loss_fn() #初始化损失函数
        self.train_data_loader, self.train_iter = aggregate_user_data(self.data, args.dataset, self.ensemble_batch_size) #聚合用户数据(但不应该对用户数据进行聚合呀？)
        self.generative_optimizer = torch.optim.Adam(
            params=self.generator_model_glob.parameters(),
            lr=self.ensemble_lr, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=self.weight_decay, amsgrad=False)
        self.discriminator_optimizer = torch.optim.Adam(
            params=self.discriminator_model_glob.parameters(),
            lr=self.ensemble_lr, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=self.weight_decay, amsgrad=False)
        self.generator_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.generative_optimizer, gamma=0.98)
        self.discriminator_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.discriminator_optimizer, gamma=0.98)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.ensemble_lr, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=0, amsgrad=False)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.002)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.98)

        #### creating users ####
        self.users = []
        for i in range(total_users):
            id, train_data, test_data=read_user_data(i, self.data, dataset=args.dataset,)
            self.total_train_samples+=len(train_data)
            self.total_test_samples += len(test_data)
            #id, train, test=read_user_data(i, data, dataset=args.dataset)
            user=UserFedHAG(
                args, id, model, self.generator_model_glob,
                train_data, test_data, self.latent_layer_idx,
                use_adam=self.use_adam)
            self.users.append(user)
        print("Number of Train/Test samples:", self.total_train_samples, self.total_test_samples)
        print("Data from {} users in total.".format(total_users))
        print("Finished creating FedAvg server.")

    def train(self, args):
        #### pretraining
        w_glob_keys = []
        loss_train = []
        w_glob = {}
        # 生成器和判别器全局模型参数
        g_glob = {}
        d_glob = {}
        lens = np.ones(args.num_users)
        w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))
        generator_model_glob = self.generator_model_glob.to(args.device)
        discriminator_model_glob = self.discriminator_model_glob.to(args.device)
        net_glob= self.model.to(args.device)
        net_glob.train()#11111
        loss_local_full = [[] for _ in range(args.num_users)]
        total_num_layers = len(net_glob.state_dict().keys())#1111
    
        w_locals = {}#11111
        for user in range(args.num_users):#1111111
            w_local_dict = {}
            for key in net_glob.state_dict().keys():
                w_local_dict[key] = net_glob.state_dict()[key]
            w_locals[user] = w_local_dict
        
        mean_train = torch.zeros(args.dim_latent)
        var_train = torch.zeros(args.dim_latent)
        net_preprocs = {}
        user_data={}
        for user_idx, user in enumerate(self.users): 
            net_preproc = get_preproc_model(args, dim_in=args.dim_latent, dim_out=args.dim_latent)
            net_preprocs[user_idx]=(net_preproc)
            id = self.data[0][user_idx]
            train_data=self.data[2][id]
            X_train, y_train = convert_data(train_data['x'], train_data['y'], dataset=self.dataset)
            pca = PCA(n_components=args.dim_latent)
            X_pca = pca.fit_transform(X_train)
            X_pca = torch.tensor(X_pca, dtype=torch.float32)
            dataset = TensorDataset(X_pca, y_train)
            user_data[user_idx] = DataLoader(dataset=dataset, batch_size=args.preproc_batch_size, drop_last = True, shuffle=True)
            mean_train+= torch.mean(X_pca, dim=0)  # 每一维的均值
            var_train+= torch.var(X_pca, dim=0) 
        total_mean_train=mean_train/args.num_users
        net_glopreproc = get_preproc_model(args, dim_in=args.dim_latent, dim_out=args.dim_latent)
        prior = Prior([args.dim_latent],total_mean_train)
        train_preproc(net_preprocs, user_data, prior,
                  n_epochs=args.align_epochs,
                  args=args,
                  verbose=True)
        net_glopreproc=aggregate_models(net_preprocs, user_data, net_glopreproc, args)
        for glob_iter in range(self.num_glob_iters):
            print("\n\n-------------Round number: ",glob_iter, " -------------\n\n")
            loss_locals = []
            total_len=0
            #im_out_list = []
            self.selected_users=self.select_users(glob_iter, self.num_users, return_idx=True)#选择用户为10个

            # 记录所有用户的本地模型参数之和
            g_locals = None
            d_locals = None

            # 记录选择的用户数量
            num_selected_users = len(self.selected_users)

            if not self.local:
                self.send_parameters(mode=self.mode)# broadcast averaged prediction model
            self.evaluate(net_glob,net_glopreproc) #输出的是average glocal accurancy，Loss
            self.send_logits()
            #chosen_verbose_user = np.random.randint(0, len(self.users))
            self.timestamp = time.time() # log user-training start time
            for userid,user in enumerate(self.selected_users): # allow selected users to train
                #verbose= user_id == chosen_verbose_user
                # perform regularization using generated samples after the first communication round
                
                local = Het_LocalUpdate(args=args, dataset= user_data[userid],current_iter=glob_iter)
                local.update_prior = args.update_prior 

                net_local = copy.deepcopy(net_glob)
                w_local = net_local.state_dict()
                for k in w_locals[userid].keys():
                    if k not in w_glob_keys:
                        w_local[k] = w_locals[userid][k]
                net_local.load_state_dict(w_local)
            
                last = glob_iter == args.num_glob_iters
            
                w_local, w_prelocal,loss, indd, last_loss,loss_w, g_local, d_local = local.train(glob_iter,
                                                  # local.train参数
                                                  net=net_local.to(args.device), 
                                                  net_preproc=net_glopreproc.to(args.device),
                                                  w_glob_keys=w_glob_keys, 
                                                  # CGAN参数
                                                  g_glob = g_glob,
                                                  d_glob = d_glob,
                                                  #######
                                                  # 中间参数
                                                  user = user,
                                                  #######
                                                  # user.train参数
                                                  global_mean = self.global_mean, 
                                                  global_variance = self.global_variance,
                                                  personalized=self.personalized,
                                                  regularization= glob_iter > 0,
                                                  #######
                                                  prior=prior,
                                                  lr=args.lr, last=last
                                                  )
                net_preprocs[userid]=w_prelocal
                loss_locals.append(copy.deepcopy(loss))
                total_len += lens[userid]
                loss_local_full[userid] = loss_local_full[userid] + copy.deepcopy(last_loss)
                print(f"User {userid}, Epoch {glob_iter}, Loss: {loss_w:.4f}")
                if len(w_glob) == 0:
                # first iteration 
                    w_glob = copy.deepcopy(w_local)
                    for k,key in enumerate(net_glob.state_dict().keys()):
                        #key_full = '1.' + key
                        w_glob[key] = w_glob[key]*lens[userid]
                        w_locals[userid][key] = w_local[key]
                else:
                    for k,key in enumerate(net_glob.state_dict().keys()):

                        if key in w_glob_keys:
                            w_glob[key] += w_local[key]*lens[userid]

                        w_locals[userid][key] = copy.deepcopy(w_local[key])

                # # 生成器模型聚合准备，简单计算平均值
                if g_locals is None:
                # first iteration 
                    g_locals = {key: g_local[key].clone() for key in g_local}
                else:
                    for key in g_locals:
                        g_locals[key] += g_local[key]

                # # 判别器模型聚合准备，简单计算平均值
                if d_locals is None:
                # first iteration 
                    d_locals = {key: d_local[key].clone() for key in d_local}
                else:
                    for key in d_locals:
                        d_locals[key] += d_local[key]

            #curr_timestamp = time.time() # log  user-training end time
            #train_time = (curr_timestamp - self.timestamp) / len(self.selected_users)
            #self.metrics['user_train_time'].append(train_time)

            # 生成器聚合，计算生成器模型参数的平均值
            for key in g_locals:
                g_glob[key] = torch.div(g_locals[key], num_selected_users)
            # 将聚合后的参数更新到全局模型
            generator_model_glob.load_state_dict(g_glob)

            # 判别器聚合，计算生成器模型参数的平均值
            for key in d_locals:
                d_glob[key] = torch.div(d_locals[key], num_selected_users)
            # 将聚合后的参数更新到全局模型
            discriminator_model_glob.load_state_dict(d_glob)

            self.aggregate_logits()

            
            
        
                # lens are the weigth of local model when doing averages
                # summing all the weights of shared global models
            
                #times_in.append( time.time() - start_in )
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
            if args.num_glob_iters != iter:
                net_glob.load_state_dict(w_glob)

            
            #self.model.loat_state_dict(w_glob)
            # self.model= net_glob.state_dict()
            # for k in w_glob_keys:
            #     w_local[k] = w_glob[k]
            # if args.num_glob_iters != iter:
            #     net_glob.load_state_dict(w_glob)

            prior.mu = prior.mu_temp/ prior.n_update
            prior.init_mu_temp()        
            
            if args.align_epochs_altern >0:
   
                prior.mu.requires_grad_(False)
                train_preproc(net_preprocs,user_data,prior,
                              n_epochs=args.align_epochs_altern,
                              args=args,
                              verbose=True
                              )
            
            net_glopreproc=aggregate_models(net_preprocs, user_data, net_glopreproc, args)

            avg_train_mse, avg_train_rmse, train_loss, avg_train_mae, avg_train_r2 = het_test_img_local_all(net_glob, net_glopreproc, args,
                                                         user_data,
                                                        w_glob_keys=w_glob_keys,
                                                        w_locals=w_locals,indd=indd)
    
            # for algs which learn a single global model, these are the local accuracies (computed using the locally updated versions of the global model at the end of each round)
            print('Round {:3d}, Train loss: {:.3f}, Train MSE: {:.3f}, Train RMSE: {:.3f}, Train MAE: {:.3f}, Train R2-score: {:.2f}'.format(
                        glob_iter, train_loss, avg_train_mse, avg_train_rmse, avg_train_mae, avg_train_r2))
            #self.evaluate(net_preprocs)
            # if self.personalized:
            #     self.evaluate_personalized_model(net_glob,net_preprocs)

            self.aggregate_parameters()
        self.save_results(args)
        self.save_model()


    
    def aggregate_logits(self, selected=True):
        sum_means = 0
        sum_variances = 0
        users = self.selected_users if selected else self.users

        for user in users:
            mean, variance = user.RegressionTracker.avg_and_var()  # 获取均值和方差
            sum_means += mean
            sum_variances += variance

        # 计算全局的均值和方差
        self.global_mean = sum_means / len(users)
        self.global_variance = sum_variances / len(users)

    def send_logits(self):
        if self.global_mean is None or self.global_variance is None:return
        for user in self.selected_users:
            user.global_mean = self.global_mean.clone().detach()
            user.global_variance = self.global_variance.clone().detach()

    def get_label_weights(self):
        label_weights = []
        qualified_labels = []
        for label in range(self.unique_labels):
            weights = []
            for user in self.selected_users:
                weights.append(user.label_counts[label])
            if np.max(weights) > MIN_SAMPLES_PER_LABEL:
                qualified_labels.append(label)
            # uniform
            label_weights.append( np.array(weights) / np.sum(weights) )
        label_weights = np.array(label_weights).reshape((self.unique_labels, -1))
        return label_weights, qualified_labels

    def visualize_images(self, generator, glob_iter, repeats=1):
        """
        Generate and visualize data for a generator.
        """
        os.system("mkdir -p images")
        path = f'images/{self.algorithm}-{self.dataset}-iter{glob_iter}.png'
        y=self.available_labels
        y = np.repeat(y, repeats=repeats, axis=0)
        y_input=torch.tensor(y)
        generator.eval()
        images=generator(y_input, latent=False)['output'] # 0,1,..,K, 0,1,...,K
        images=images.view(repeats, -1, *images.shape[1:])
        images=images.view(-1, *images.shape[2:])
        save_image(images.detach(), path, nrow=repeats, normalize=True)
        print("Image saved to {}".format(path))
