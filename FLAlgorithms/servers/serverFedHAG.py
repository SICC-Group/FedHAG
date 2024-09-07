from FLAlgorithms.users.userFedHAG import UserFedHAG
#from FLAlgorithms.users.userpFedGen import RegressionTracker
from FLAlgorithms.servers.serverbase import Server
from FLAlgorithms.trainmodel.RKLDivLoss import RKLDivLoss
from FLAlgorithms.users.userpFedGen import RegressionTracker
from utils.model_utils import create_model,convert_data
from utils.model_utils import read_data, read_user_data, aggregate_user_data, create_generative_model, create_discriminator_model
from Het_Update import Het_LocalUpdate, het_test_img_local_all, train_preproc
from Het_Nets import get_reg_model, get_preproc_model
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

        times = []
        times_in = []
        lens = np.ones(args.num_users)
        w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))
        generator_model_glob = self.generator_model_glob.to(args.device)
        discriminator_model_glob = self.discriminator_model_glob.to(args.device)
        net_glob= self.model.to(args.device)
        net_glob.train()#11111
        loss_local_full = [[] for _ in range(args.num_users)]
        total_num_layers = len(net_glob.state_dict().keys())#1111
        print(total_num_layers)
        print(net_glob.state_dict().keys())
    
        w_locals = {}#11111
        for user in range(args.num_users):#1111111
            w_local_dict = {}
            for key in net_glob.state_dict().keys():
                w_local_dict[key] = net_glob.state_dict()[key]
            w_locals[user] = w_local_dict

        
        prior = Prior([args.dim_latent],args.mean_target_variance)
        net_preprocs = {}
        user_data={}
        for user_idx, user in enumerate(self.users): 
            data_point = user.trainloader.dataset[0]
            # Check if data_point is a tuple or list (as it seems to be in your case)
            if isinstance(data_point, (tuple, list)):
               # Access the first element of the tuple (which is likely the tensor)
               tensor = data_point[0]
               # Get the shape of the tensor
               shape = tensor.shape
               # Now you can check the dimensions
               if not isinstance(shape[0], int) or shape[0] <= 0:
            #        print(f"Invalid shape: {shape}")   
            # if not isinstance(user.trainloader.dataset[0].tensors.shap[1], int) or user.trainloader.dataset[0].tensors.shap[1] <= 0:
                raise ValueError(f"The feature number for user {user} is not a valid integer")
                   # 使用 f_num[user] 作为 dim_in 参数创建模型
            net_preproc = get_preproc_model(args, dim_in=shape[0], dim_out=args.dim_latent)
            net_preprocs[user_idx]=(net_preproc)
            id = self.data[0][user_idx]
            train_data=self.data[2][id]
            X_train, y_train = convert_data(train_data['x'], train_data['y'], dataset=self.dataset)
            dataset = TensorDataset(X_train, y_train)
            user_data[user_idx] = DataLoader(dataset=dataset, batch_size=args.preproc_batch_size, drop_last = True, shuffle=True)
        train_preproc(net_preprocs, user_data, prior,
                  n_epochs=args.align_epochs,
                  args=args,
                  verbose=True)
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
            self.evaluate(net_glob,net_preprocs) #输出的是average glocal accurancy，Loss
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
            
                w_local, loss, indd, last_loss,loss_w, g_local, d_local = local.train(
                                                  # local.train参数
                                                  net=net_local.to(args.device), 
                                                  net_preproc=net_preprocs[userid].to(args.device),
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
                print(f"User {userid}, Epoch {glob_iter}, Loss: {loss_w:.4f}")
                #X_train, y_train = convert_data(user.trainloader.dataset['x'], user.trainloader.dataset['y'], dataset=self.dataset)
                # net_preprocTD = net_preprocs[userid].to(args.device)
                # im_out = net_preprocTD(user_data[userid].dataset.tensors[0])
                # #im_out_list.append(im_out.detach())  # 记录 im_out
                # mapdataset = TensorDataset(im_out.detach(), user_data[userid].dataset.tensors[1])
                # #mapdataset_nums = len(mapdataset)
                # maptrainloader = DataLoader(mapdataset, self.batch_size, shuffle=True, drop_last=True)
                # iter_maptrainloader = iter(maptrainloader)
                # w_local=user.train(
                #          net_local.to(args.device), maptrainloader,iter_maptrainloader,glob_iter, self.global_mean, self.global_variance,prior,
                #          personalized=self.personalized, 
                #          early_stop=self.early_stop,
                #          verbose=True and glob_iter > 0,
                #          regularization= glob_iter > 0 )#计算本地模型的总损失，包括预测损失、教师损失和潜在损失。这一个过程只更新本地模型参数
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

            loss_locals.append(copy.deepcopy(loss))
            total_len += lens[userid]
            loss_local_full[userid] = loss_local_full[userid] + copy.deepcopy(last_loss)
            
        
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

            # if args.align_epochs_altern >0:
   
                # prior.mu.requires_grad_(False)
                # train_preproc(net_preprocs,user_data,prior,
                #               n_epochs=args.align_epochs_altern,
                #               args=args,
                #               verbose=True
                #               )
            
                
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
                        glob_iter, train_loss, avg_train_mse, avg_train_rmse, avg_train_mae, avg_train_r2))
            #self.evaluate(net_preprocs)
            if self.personalized:
                self.evaluate_personalized_model(net_glob,net_preprocs)

            # 聚合生成器和判别器的模型
            # 聚合生成器模型参数
            #aggregated_generator_state_dict = self.aggregate_models(g_local, self.num_users)
            # 聚合判别器模型参数
            #aggregated_discriminator_state_dict = self.aggregate_models(d_local, self.num_users)
            # 将聚合后的模型参数加载到全局模型中,记得修改全局生成器和判别器的名称
            #global_generator.load_state_dict(aggregated_generator_state_dict)
            #global_discriminator.load_state_dict(aggregated_discriminator_state_dict)

            # self.timestamp = time.time() # log server-agg start time

            # self.train_generator(
            #     net_glob.to(args.device),self.batch_size, self.global_mean, self.global_variance,prior, im_out_list=im_out_list,  # 传递 im_out_list
            #     epoches=self.ensemble_epochs // self.n_teacher_iters,
            #     latent_layer_idx=self.latent_layer_idx,
            #     verbose=True
            # ) #更新生成器模型，不更新本地模型参数。其中损失函数包括，教师损失（生成器的输出作为预测模型的输入，得到的预测标签与随机生成标签之间的差别），多样性损失（生成样本与噪声之间的差别）
            
            # self.aggregate_parameters()
            # curr_timestamp=time.time()  # log  server-agg end time
            # agg_time = curr_timestamp - self.timestamp
            # self.metrics['server_agg_time'].append(agg_time)
            #if glob_iter  > 0 and glob_iter % 20 == 0 and self.latent_layer_idx == 0:
                #self.visualize_images(self.generative_model, glob_iter, repeats=10)
            self.aggregate_parameters()
        self.save_results(args)
        self.save_model()


    # 改成本地训练的函数,移到local.train中
    # # def local_train_generator(self, prior, im_out, target, latent_layer_idx=0):
    #     """
    #     Learn a generator that find a consensus latent representation z, given a label 'y'.
    #     :param batch_size:
    #     :param epoches:
    #     :param latent_layer_idx: if set to -1 (-2), get latent representation of the last (or 2nd to last) layer.
    #     :param verbose: print loss information.
    #     :return: Do not return anything.
    #     """
    #     #self.generative_model.train()
    #     #self.model.eval()
    #     #self.generative_regularizer.train()
    #     #self.label_weights, self.qualified_labels = self.get_label_weights()
    #     #TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS, STUDENT_LOSS2 = 0, 0, 0, 0
    #     #rkl=RKLDivLoss()

    #     def update_generator_():
    #         #num_features=1
    #         #RegressionTracker2=RegressionTracker(num_features)
    #         #RegressionTracker3=RegressionTracker(num_features)
    #         self.generative_model.train()
    #         self.discriminator_model.train()
    #         #student_model.eval()
    #         #self.latent_model.eval()
    #         #for i in range(n_iters):
    #         self.discriminator_optimizer.zero_grad()
    #         self.generative_optimizer.zero_grad()

    #         # 改成y = target
    #         #y = torch.randn(batch_size) * global_variance.sqrt() + global_mean
    #         y = target

    #         #y_input=y
    #         ## feed to generator
    #         gen_result=self.generative_model(y, latent_layer_idx=latent_layer_idx, prior=prior,verbose=True)
    #         # get approximation of Z( latent) if latent set to True, X( raw image) otherwise
    #         gen_output, eps=gen_result['output'], gen_result['eps']

    #         # 这里对real_score的计算以及对im_out的处理都要再次进行修改
    #         # 计算总元素数量
    #         total_elements = im_out.numel()

    #         # 创建目标维度的张量
    #         target_size = (32, 32)
    #         target_elements = target_size[0] * target_size[1]

    #         if total_elements >= target_elements:
    #             # 截取数据并重塑为目标维度
    #             im_out_preprocessed = im_out.view(-1)[:target_elements].reshape(target_size)
    #         else:
    #             # 填充数据并重塑为目标维度
    #             im_out_preprocessed = torch.zeros(target_size)
    #             im_out_preprocessed[:im_out.size(0), :im_out.size(1)] = im_out
    #         # 进行判别器的训练
    #         d_loss_real = self.discriminator_model(im_out_preprocessed, y)
    #         #### im_out的处理需要修改
    #         # Pass generated data through discriminator
    #         d_loss_fake = self.discriminator_model(gen_output.detach(), y)
    #         # Compute the loss for the discriminator
    #         #d_loss_real = torch.mean(torch.nn.functional.binary_cross_entropy_with_logits(real_score, torch.ones_like(real_score)))
    #         #d_loss_fake = torch.mean(torch.nn.functional.binary_cross_entropy_with_logits(fake_score, torch.zeros_like(fake_score)))
    #         d_loss = 0.5 * (d_loss_real + d_loss_fake)
    #         d_loss.backward()
    #         self.discriminator_optimizer.step()

    #         ##### get losses ####
    #         # decoded = self.generative_regularizer(gen_output)
    #         # regularization_loss = beta * self.generative_model.dist_loss(decoded, eps) # map generated z back to eps
    #         #diversity_loss=self.generative_model.diversity_loss(eps, gen_output)  # encourage different outputs

    #         # Again pass generated data through discriminator
    #         g_loss = self.discriminator_model(gen_output, y)
    #         # Compute the generator loss based on discriminator feedback
    #         #g_loss_adv = torch.mean(torch.nn.functional.binary_cross_entropy_with_logits(fake_score, torch.ones_like(fake_score)))

    #         ######### get teacher loss ############
    #         # teacher_loss=0
    #         # mean_total=0
    #         # variance_total=0
    #         # mean_avg=0
    #         # variance_avg=0
    #         # teacher_logit=0
    #         # y_input=y.unsqueeze(-1)
    #         # for user_idx, user in enumerate(self.selected_users):
    #         #     #user.latent_model.eval()
    #         #     user.model.eval()
    #         #     #weight=self.label_weights[y][:, user_idx].reshape(-1, 1)
    #         #     #expand_weight=np.tile(weight, (1, self.unique_labels))
    #         #     #gen_output = gen_output.permute(1, 0)
    #         #     user_result_given_gen=user.model(gen_output,start_layer_idx=1)['output']
    #         #     RegressionTracker2.update(user_result_given_gen.detach())
    #         #     mean, variance = RegressionTracker2.avg_and_var()
    #         #     mean_total+=mean
    #         #     variance_total+=variance
                
    #         #     #user_output_logp_=F.log_softmax(user_result_given_gen['logit'], dim=1)
    #         #     teacher_loss_=torch.mean(self.generative_model.dist_loss(user_result_given_gen, y_input))
    #         #     teacher_loss+=teacher_loss_
    #         #     #teacher_logit+=user_result_given_gen['logit'] * torch.tensor(expand_weight, dtype=torch.float32)
    #         # mean_avg=mean_total/len(self.selected_users)
    #         # variance_avg=variance_total/len(self.selected_users)
    #         # ######### get student loss ############
    #         # student_output=student_model(gen_output,start_layer_idx=1)['output']
    #         # RegressionTracker3.update(student_output.detach())
    #         # mean1, variance1 = RegressionTracker3.avg_and_var()
    #         # student_loss=rkl(mean1, variance1, mean_avg, variance_avg)
    #         # if self.ensemble_beta > 0:
    #         #     loss=self.ensemble_alpha * teacher_loss - self.ensemble_beta * student_loss + self.ensemble_eta * diversity_loss + g_loss_adv
    #         # else:
    #         #     loss=self.ensemble_alpha * teacher_loss + self.ensemble_eta * diversity_loss + g_loss_adv
    #         g_loss.backward()
    #         self.generative_optimizer.step()
    #         #TEACHER_LOSS += self.ensemble_alpha * teacher_loss#(torch.mean(TEACHER_LOSS.double())).item()
    #         #STUDENT_LOSS += self.ensemble_beta * student_loss#(torch.mean(student_loss.double())).item()这一项一直没有用
    #         #DIVERSITY_LOSS += self.ensemble_eta * diversity_loss#(torch.mean(diversity_loss.double())).item()
    #         #return TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS
    #         return d_loss, g_loss

    #     #for i in range(epoches):
    #     d_loss, g_loss=update_generator_()

    #     # TEACHER_LOSS = TEACHER_LOSS.detach().numpy() / (self.n_teacher_iters * epoches)
    #     # STUDENT_LOSS = STUDENT_LOSS.detach().numpy() / (self.n_teacher_iters * epoches)
    #     # DIVERSITY_LOSS = DIVERSITY_LOSS.detach().numpy() / (self.n_teacher_iters * epoches)
    #     # info="Generator: Teacher Loss= {:.4f}, Student Loss= {:.4f}, Diversity Loss = {:.4f}, ". \
    #     #     format(TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS)
    #     # if verbose:
    #     #     print(info)
    #     info="Generator Loss= {:.4f}, Discriminator Loss= {:.4f}, ". \
    #          format(g_loss, d_loss)
    #     print(info)
    #     self.generative_lr_scheduler.step()
    #     self.discriminator_lr_scheduler.step()

    # # 聚合生成器模型
    # def aggregate_models(client_models, num_clients):
    #     aggregated_model = {}
    #     for key in client_models[0].keys():
    #         aggregated_model[key] = sum([client_models[i][key] for i in range(num_clients)]) / num_clients
    #     return aggregated_model
    
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
