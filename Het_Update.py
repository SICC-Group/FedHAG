# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/models/Update.py
# credit: Paul Pu Liang

# For MAML (PerFedAvg) implementation, code was adapted from https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch/blob/master/few_shot_learning_system.py
# credit: Antreas Antoniou

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import math
import torch.linalg as linalg

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import copy
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from FLAlgorithms.users.userbase import User
from utils.model_utils import read_data, read_user_data, aggregate_user_data, create_generative_model, create_discriminator_model
from utils.model_utils import get_dataset_name, RUNCONFIGS



def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad

def dist_torch(x1,x2):
    x1p = x1.pow(2).sum(1).unsqueeze(1)
    x2p = x2.pow(2).sum(1).unsqueeze(1)
    prod_x1x2 = torch.mm(x1,x2.t())
    distance = x1p.expand_as(prod_x1x2) + x2p.t().expand_as(prod_x1x2) -2*prod_x1x2
    return distance #/x1.size(0)/x2.size(0) 


def calculate_mmd(X,Y):
    def my_cdist(x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)
    
    def gaussian_kernel(x, y, gamma=[0.0001,0.001, 0.01, 0.1, 1, 10, 100]):
        D = my_cdist(x, y)
        K = torch.zeros_like(D)
    
        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))
    
        return K
    

    Kxx = gaussian_kernel(X, X).mean()
    Kyy = gaussian_kernel(Y, Y).mean()
    Kxy = gaussian_kernel(X, Y).mean()
    return Kxx + Kyy - 2 * Kxy


def calculate_2_wasserstein_dist(X, Y):
    '''
    Calulates the two components of the 2-Wasserstein metric:
    The general formula is given by: d(P_X, P_Y) = min_{X, Y} E[|X-Y|^2]
    For multivariate gaussian distributed inputs z_X ~ MN(mu_X, cov_X) and z_Y ~ MN(mu_Y, cov_Y),
    this reduces to: d = |mu_X - mu_Y|^2 - Tr(cov_X + cov_Y - 2(cov_X * cov_Y)^(1/2))
    Fast method implemented according to following paper: https://arxiv.org/pdf/2009.14075.pdf
    Input shape: [b, n] (e.g. batch_size x num_features)
    Output shape: scalar
    '''


    if X.shape != Y.shape:
        raise ValueError("Expecting equal shapes for X and Y!")

    # the linear algebra ops will need some extra precision -> convert to double
    X, Y = X.transpose(0, 1).double(), Y.transpose(0, 1).double()  # [n, b]
    mu_X, mu_Y = torch.mean(X, dim=1, keepdim=True), torch.mean(Y, dim=1, keepdim=True)  # [n, 1]
    n, b = X.shape
    fact = 1.0 if b < 2 else 1.0 / (b - 1)

    # Cov. Matrix
    E_X = X - mu_X
    E_Y = Y - mu_Y
    cov_X = torch.matmul(E_X, E_X.t()) * fact  # [n, n]
    cov_Y = torch.matmul(E_Y, E_Y.t()) * fact

    # calculate Tr((cov_X * cov_Y)^(1/2)). with the method proposed in https://arxiv.org/pdf/2009.14075.pdf
    # The eigenvalues for M are real-valued.
    C_X = E_X * math.sqrt(fact)  # [n, n], "root" of covariance
    C_Y = E_Y * math.sqrt(fact)
    M_l = torch.matmul(C_X.t(), C_Y)
    M_r = torch.matmul(C_Y.t(), C_X)
    M = torch.matmul(M_l, M_r)
    # print(M)
    S = linalg.eigvals(M+1e-6) + 1e-15  # add small constant to avoid infinite gradients from sqrt(0)
    sq_tr_cov = S.sqrt().abs().sum()

    # plug the sqrt_trace_component into Tr(cov_X + cov_Y - 2(cov_X * cov_Y)^(1/2))
    trace_term = torch.trace(cov_X + cov_Y) - 2.0 * sq_tr_cov  # scalar

    # |mu_X - mu_Y|^2
    diff = mu_X - mu_Y  # [n, 1]
    mean_term = torch.sum(torch.mul(diff, diff))  # scalar

    # put it together
    return (trace_term + mean_term).float()




# def wass_loss(net_projector,data, label, prior,optimize_projector=True,distance='wd'):
#     loss = 0
#     present_label = torch.unique(label)
#     for curr_label in present_label:                          
#         ind_l = torch.where(label==curr_label)[0]
#         #print(ind_l.shape,label)
#         if ind_l.shape[0]>0:
#             if optimize_projector:
#                 #print(data[ind_l].shape)
#                 out = net_projector(data[ind_l])
    
#                 with torch.no_grad():
#                     mean_t = prior.mu[curr_label].detach()
#                     var_t = prior.logvar[curr_label].detach()
                        
#                 prior_samples  = prior.sampling_gaussian(out.shape[0], mean_t, var_t) 
#                 if distance == 'wd':
#                     loss +=  calculate_2_wasserstein_dist(out,prior_samples)
#                 elif distance == 'mmd':       
#                     loss += calculate_mmd(out, prior_samples)
    
#             else:
#                 # we optimize the local means of the priors, so we work on local
#                 # vectors
#                 set_requires_grad(net_projector,requires_grad=False)
#                 with torch.no_grad():
#                     out = net_projector(data[ind_l])
    
    
#                 prior_samples  = prior.sampling_gaussian(out.shape[0], prior.mu_local[curr_label], prior.logvar[curr_label]) 
    
#                 if distance == 'wd':
#                     loss +=  calculate_2_wasserstein_dist(out,prior_samples)
#                 elif distance == 'mmd':
#                     loss += calculate_mmd(out, prior_samples)
#     return loss

def wass_loss(net_projector, data, target, prior, optimize_projector=True, distance='wd'):
    """
    Calculate the Wasserstein or MMD loss for a regression task.

    Parameters:
    - net_projector: The regression model (or projector).
    - data: Input data.
    - label: Ground truth labels (though in regression, labels are continuous targets).
    - prior: The Prior object to sample from.
    - optimize_projector: Whether to optimize the projector.
    - distance: Type of distance metric to use ('wd' for Wasserstein distance or 'mmd' for MMD).

    Returns:
    - loss: Calculated loss value.
    """
    loss = torch.tensor(0.0, requires_grad = True)
     
    out = net_projector(data)
    # print("*",out.shape)
    # print(out)
        # Sample from the prior using the regression outputs
        #mean_t = prior.mu.detach()  # Assuming we use the first component as a reference
        #var_t = prior.logvar.detach()
    mean_t = prior.mu  # Assuming we use the first component as a reference
    var_t = prior.logvar
            
    prior_samples = prior.sampling_gaussian(out.shape[0], mean_t, var_t)
    # print("#",prior_samples.shape)
    # print(prior_samples)
    if distance == 'wd':
        loss_dist = calculate_2_wasserstein_dist(out, prior_samples)
        loss = torch.add(loss, loss_dist, alpha = 1)
    elif distance == 'mmd':
        loss_dist = calculate_mmd(out, prior_samples)
        loss = torch.add(loss, loss_dist, alpha = 1)

    return loss


def train_preproc(net_preprocs, user_data, prior,n_epochs,args=None,verbose=True):
    
    prior.mu = prior.mu.to(args.device)  
    prior.logvar = prior.logvar.to(args.device)  
    if args.device == 'cuda':
        pin_memory = True
    else:
        pin_memory = False

    # for each user, optimize the loss
    idxs_users = np.arange(args.num_users)

    for ind, idx in enumerate(idxs_users):
        # data_adapt = DataLoader(dataset=dataset_train, batch_size=args.align_bs, shuffle=True
        #                             ,drop_last=False,pin_memory=pin_memory,
        #                             num_workers=args.num_workers)
        data_adapt = user_data[idx]
        # data_iter = iter(dataset_train)
        net_projector =  copy.deepcopy(net_preprocs[idx].to(args.device))
        set_requires_grad(net_projector, requires_grad=True)
        
        optimizer_preproc = torch.optim.Adam(net_projector.parameters(),lr=args.align_lr)

        for itm in range(n_epochs):
            loss_tot = 0
            # try:
            #     for _ in range(len(dataset_train)):
            #         data, target =next(data_iter)
            
            # except StopIteration:
            #     data_iter = iter(dataset_train)
            #     continue
            
            # data = data.to(args.device)
            # target = target.to(args.device)
            # optimizer_preproc.zero_grad()
            # net_projector.zero_grad()
            # loss = wass_loss(net_projector, data, target, prior,optimize_projector=True,distance=args.distance)
            # loss.backward()
            # optimizer_preproc.step()
            # loss_tot +=loss.item()
            for data, target in data_adapt:
                data = data.to(args.device)  
                target = target.to(args.device) 
                target=target.view(-1, 1)   
                # data.requires_grad_(True)  
                # target.requires_grad_(True)            
                optimizer_preproc.zero_grad()
                net_projector.zero_grad()
                loss = wass_loss(net_projector, data, target, prior,optimize_projector=True,distance=args.distance)
                loss.backward()
                optimizer_preproc.step()

                loss_tot +=loss.item()
            if verbose :
                print(f"User {ind}, Epoch {itm}, Loss: {loss_tot:.4f}")
        set_requires_grad(net_projector, requires_grad=False)
        net_preprocs[idx] = copy.deepcopy(net_projector)



class Het_LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, indd=None,
                 mean_target=None,current_iter= 1000):
        self.args = args
        #移除交叉熵损失，换成均方误差损失
        #self.loss_func = nn.CrossEntropyLoss()   
        self.loss_func = nn.MSELoss() 
        self.ldr_train = DataLoader(dataset, batch_size=self.args.local_bs, shuffle=True,
                                    pin_memory=True,
                                    num_workers=args.num_workers)
        #self.ldr_train = dataset
        self.dataset = dataset
        self.idxs = idxs
        self.indd = indd
        self.update_prior = args.update_prior > 0
        self.update_net_preproc = args.update_net_preproc > 0
        self.update_global_representation = current_iter > args.start_optimize_rep

        self.ensemble_lr = 1e-4
        self.weight_decay = 1e-2
    

        # 生成器和判别器
        self.generator_model = create_generative_model(args.dataset, args.algorithm, args.embedding)#生成器模型
        self.discriminator_model = create_discriminator_model(args.dataset, args.algorithm, args.embedding) # 判别器模型
        self.latent_layer_idx = self.generator_model.latent_layer_idx
        self.generator_optimizer = torch.optim.Adam(
            params=self.generator_model.parameters(),
            lr=self.ensemble_lr, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=self.weight_decay, amsgrad=False)
        self.discriminator_optimizer = torch.optim.Adam(
            params=self.discriminator_model.parameters(),
            lr=self.ensemble_lr, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=self.weight_decay, amsgrad=False)
        self.generative_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.generator_optimizer, gamma=0.98)
        self.discriminator_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.discriminator_optimizer, gamma=0.98)
        
    def train(self, net, net_preproc, w_glob_keys, g_glob, d_glob, user, global_mean, global_variance, personalized, regularization, last=False, dataset_test=None, 
              prior=None,ind=-1, idx=-1, lr=0.1):
        bias_p=[]
        weight_p=[]
        #net_projector =  copy.deepcopy(net.to(self.args.device))
        for name, p in net.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        # optimizer = torch.optim.Adam(
        # [     
        #     {'params': weight_p, 'weight_decay':lr},
        #     {'params': bias_p, 'weight_decay':0}
        # ],
        # lr=lr
        # )        
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        local_eps = self.args.local_epochs
        if last:
            local_eps =  max(10,local_eps-self.args.local_rep_ep)
        if self.update_global_representation:
            head_eps = local_eps-self.args.local_rep_ep
        else:
            head_eps = local_eps
        epoch_loss = []
        num_updates = 0
        mu_local = nn.Parameter(prior.mu.clone(), requires_grad=True)
        optim_mean = torch.optim.Adam([mu_local],lr=0.001)
        net.to(self.args.device)
        net_preproc.to(self.args.device)
        mu_local = mu_local.to(self.args.device)
        prior.mu_temp = prior.mu_temp.to(self.args.device)  
        optimizer_preproc = torch.optim.Adam(net_preproc.parameters(),lr=self.args.align_lr)
        # 10
        for iter in range(local_eps):
            net_preproc.train()
            net.train()
            #net_projector.eval()
            batch_loss = []
            for batch_idx, (data, target) in enumerate(self.dataset):
                # first 9 round
                if (iter < head_eps ) or last or not w_glob_keys:
                    # update net_preproc parameter
                    if self.update_net_preproc:
                        set_requires_grad(net_preproc, requires_grad=True)
                    # not update net parameter
                    for name, param in net.named_parameters():
                        if name in w_glob_keys:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                #last round
                elif (iter >= head_eps ):
                    # not update net_preproc parameter
                    set_requires_grad(net_preproc, requires_grad=False)
                    # update net parameter
                    for name, param in net.named_parameters():
                        if name in w_glob_keys:
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                im_out = net_preproc(data)
                #log_probs = net(im_out)
                predictions = net(im_out.float())['output']
                #loss_regression = self.loss_func(log_probs)
                #target=target.view(-1, 1)y = y.unsqueeze(-1)
                target = target.unsqueeze(-1)
                target=target.float()
                loss_regression = self.loss_func(predictions, target)
                if self.update_net_preproc:
                    loss_W = wass_loss(net_preproc, data, target, prior,optimize_projector=True)
                else:
                    loss_W = 0
                loss_ref_dist = 0
                #present_label = torch.unique(labels)
                #回归任务不涉及类别标签，移除基于 curr_label 的循环
                # for curr_label in range(prior.mu.shape[0]):                      
                #     prior_samples  = prior.sampling_gaussian(data.shape[0], prior.mu[curr_label], prior.logvar[curr_label]) 
                #     log_probs_prior_samples = net(prior_samples)
                #     loss_ref_dist += self.loss_func(log_probs_prior_samples, curr_label*torch.ones(data.shape[0]).long().
                #                                     to(self.args.device)) 
                prior_samples = prior.sampling_gaussian(data.shape[0], prior.mu, prior.logvar)
                predictions_prior_samples = net(prior_samples.float())['output']
                loss_ref_dist = self.loss_func(predictions_prior_samples, target.float())
                num_updates += 1
                loss =  loss_regression + self.args.reg_w*loss_W + self.args.reg_reg_prior*loss_ref_dist
                batch_loss.append(loss.item())
                optimizer.zero_grad()
                optimizer_preproc.zero_grad()
                loss.backward()
                optimizer.step()
                if self.update_net_preproc:
                    optimizer_preproc.step()
                if self.update_prior:
                    prior.mu_local = mu_local
                    lossW = wass_loss(net_preproc,data, target, prior,optimize_projector=False)
                    set_requires_grad(net, requires_grad=False)
                    #log_probs = net(mu_local)  
                    predictions_probs = net(mu_local)['output']
                    #predictions_probs = predictions_probs.detach()
                    target_mean=target.mean(dim=0)
                    #labels = torch.Tensor([ii for ii in range(self.args.num_classes)]).long()
                    #labels = labels.to(self.args.device)
                    loss = self.loss_func(predictions_probs, target_mean.float()) + lossW
                    optim_mean.zero_grad()
                    loss.backward()
                    optim_mean.step()
                    for name, param in net.named_parameters():
                        if name in w_glob_keys:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True   

                # CGAN #######################
                g_local, d_local = self.local_train_generator(g_glob, d_glob, prior, im_out, target, latent_layer_idx=self.latent_layer_idx)
                ##################################

                # user.train ##################
                w_local=user.train(
                         net, im_out.detach(), target, global_mean, global_variance, regularization, prior,
                         personalized)#计算本地模型的总损失，包括预测损失、教师损失和潜在损失。这一个过程只更新本地模型参数

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        set_requires_grad(net_preproc, requires_grad=True)
        #set_requires_grad(net, requires_grad=True)
        prior.mu_temp += mu_local.detach()
        prior.n_update += 1
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.indd, epoch_loss, loss_W, g_local, d_local

    # 改成本地训练的函数,移到local.train中
    def local_train_generator(self, g_glob, d_glob, prior, im_out, target, latent_layer_idx=0):
        """
        本地训练生成器和判别器，返回g_loss与d_loss
        """

        def update_generator_():

            if(g_glob):
                # 在本地训练开始之前，从服务器获取全局模型并更新本地模型
                self.generator_model.load_state_dict(g_glob)
            if(d_glob):
                # 在本地训练开始之前，从服务器获取全局模型并更新本地模型
                self.discriminator_model.load_state_dict(d_glob)
        
            self.generator_model.train()
            self.discriminator_model.train()
            
            self.generator_optimizer.zero_grad()

            y = target

            ## feed to generator
            gen_result=self.generator_model(y, latent_layer_idx=latent_layer_idx, prior=prior,verbose=True)
            # get approximation of Z( latent) if latent set to True, X( raw image) otherwise
            gen_output, eps=gen_result['output'], gen_result['eps']
            #print("gen_output:",gen_output.shape)

            # Pass generated data through discriminator
            fake_score = self.discriminator_model(gen_output, y)

            # Again pass generated data through discriminator
            g_loss = torch.mean(torch.nn.functional.binary_cross_entropy_with_logits(fake_score, torch.ones_like(fake_score)))
            g_loss.backward()
            self.generator_optimizer.step()

            self.discriminator_optimizer.zero_grad()

            # 进行判别器的训练
            real_score = self.discriminator_model(im_out.detach(), y)
            fake_score = self.discriminator_model(gen_output.detach(), y)
            d_loss_real = torch.mean(torch.nn.functional.binary_cross_entropy_with_logits(real_score, torch.ones_like(real_score)))
            d_loss_fake = torch.mean(torch.nn.functional.binary_cross_entropy_with_logits(fake_score, torch.zeros_like(fake_score)))
            
            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            d_loss.backward()
            self.discriminator_optimizer.step()


            #TEACHER_LOSS += self.ensemble_alpha * teacher_loss#(torch.mean(TEACHER_LOSS.double())).item()
            #STUDENT_LOSS += self.ensemble_beta * student_loss#(torch.mean(student_loss.double())).item()这一项一直没有用
            #DIVERSITY_LOSS += self.ensemble_eta * diversity_loss#(torch.mean(diversity_loss.double())).item()
            #return TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS
            return d_loss, g_loss, self.generator_model.state_dict(), self.discriminator_model.state_dict()

        #
        d_loss, g_loss, g_local, d_local =update_generator_()

        # TEACHER_LOSS = TEACHER_LOSS.detach().numpy() / (self.n_teacher_iters * epoches)
        # STUDENT_LOSS = STUDENT_LOSS.detach().numpy() / (self.n_teacher_iters * epoches)
        # DIVERSITY_LOSS = DIVERSITY_LOSS.detach().numpy() / (self.n_teacher_iters * epoches)
        # info="Generator: Teacher Loss= {:.4f}, Student Loss= {:.4f}, Diversity Loss = {:.4f}, ". \
        #     format(TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS)
        # if verbose:
        #     print(info)
        info="Generator Loss= {:.4f}, Discriminator Loss= {:.4f}, ". \
                format(g_loss, d_loss)
        #print(info)
        self.generative_lr_scheduler.step()
        self.discriminator_lr_scheduler.step()

        return g_local, d_local


def het_test_img_local(net_g, net_preproc, user_data, args,idx=None,indd=None, user_idx=-1, idxs=None):
    net_g.eval()
    net_preproc.eval()
    test_loss = 0
    #correct = 0
    net_preproc.to(args.device)
    net_g.to(args.device)
    

    count = 0
    #data_loader = DataLoader(user_data, batch_size=200, shuffle=True,drop_last=False)
    data_loader = user_data
    for idx, (data, target) in enumerate(data_loader):

        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
            
        with torch.no_grad():
            prediction = net_g(net_preproc(data))
        target=target.unsqueeze(-1)
        #log_probs = net_g(net_preproc(data))
        # sum up batch loss
        #test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        batch_loss = F.mse_loss(prediction['output'], target.float(), reduction='sum').item()
        test_loss += batch_loss
        #y_pred = log_probs.data.max(1, keepdim=True)[1]
        #correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        count += data.shape[0]

        if idx==0:
            target_all = target.detach().cpu()
            y_pred_all = prediction['output'].detach().cpu()
        else:
            target_all = torch.cat((target_all,target.detach().cpu()),dim=0)
            y_pred_all = torch.cat((y_pred_all,prediction['output'].detach().cpu()),dim=0)
                
    test_loss /= count
    #accuracy = 100.00 * float(correct) / count
    #
    # bal_acc = 100*balanced_accuracy_score(target_all, y_pred_all.long())
    mse = mean_squared_error(target_all.numpy(), y_pred_all.numpy())
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(target_all.numpy(), y_pred_all.numpy())
    r2 = r2_score(target_all.numpy(), y_pred_all.numpy())
    
    
    net_g.train()
    net_preproc.train()
    
    return  mse, rmse, test_loss, mae, r2 

def het_test_img_local_all(net, net_preprocs, args, users_test_data,w_locals=None
                           ,w_glob_keys=None, indd=None,dataset_train=None,dict_users_train=None, return_all=False):
    tot = 0
    num_idxxs = args.num_users
    #acc_test_local = np.zeros(num_idxxs)
    #loss_test_local = np.zeros(num_idxxs)
    #bal_acc_test_local = np.zeros(num_idxxs)
    mse_test_local = np.zeros(num_idxxs)
    rmse_test_local = np.zeros(num_idxxs)
    mae_test_local = np.zeros(num_idxxs)
    r2_test_local = np.zeros(num_idxxs)

    for idx in range(num_idxxs):
        # net is the global network.
        # it is copied and then local layers are overwritten.
        #print(idx)
        net_local = copy.deepcopy(net)
        if w_locals is not None:
            w_local = net_local.state_dict()
            for k in w_locals[idx].keys():
                if w_glob_keys is not None and k not in w_glob_keys:
                    w_local[k] = w_locals[idx][k]
                elif w_glob_keys is None:
                    w_local[k] = w_locals[idx][k]
            net_local.load_state_dict(w_local)
        net_local.eval()
        net_preproc = net_preprocs[idx]
        net_preproc.eval()

        mse, rmse, test_loss, mae, r2= het_test_img_local(net_local,net_preproc, users_test_data[idx], args, user_idx=idx) 
        
        dataset_test = users_test_data[idx].dataset
        test_tensors = dataset_test.tensors
        n_test = test_tensors[0].shape[0]
        # n_test = users_test_data[idx].tensors[0].shape[0]
        # n_test = len(users_test_data[idx].idxs)
        # n_test = users_test_data[idx].shape[0]
        tot += n_test
        mse_test_local[idx] = mse * n_test
        rmse_test_local[idx] = rmse * n_test
        mae_test_local[idx] = mae * n_test
        r2_test_local[idx] = r2 * n_test
        #bal_acc_test_local[idx] = c*n_test

        del net_local
        net_preproc.train()
    
    avg_mse = sum(mse_test_local) / tot
    avg_rmse = sum(rmse_test_local) / tot
    avg_mae = sum(mae_test_local) / tot
    avg_r2 = sum(r2_test_local) / tot
    if return_all:
        return mse_test_local, rmse_test_local, test_loss, mae_test_local, r2_test_local
    
    return  avg_mse, avg_rmse, test_loss, avg_mae, avg_r2


