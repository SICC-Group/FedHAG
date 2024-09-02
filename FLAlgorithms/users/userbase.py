import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
import numpy as np
import copy
from FLAlgorithms.trainmodel.RKLDivLoss import RKLDivLoss
from utils.model_utils import get_dataset_name
#from torch.nn.modules.loss import RKLDivLoss
from utils.model_config import RUNCONFIGS
from FLAlgorithms.optimizers.fedoptimizer import pFedIBOptimizer
from sklearn.decomposition import PCA

class User:
    """
    Base class for users in federated learning.
    """
    def __init__(
            self, args, id, model, train_data, test_data, use_adam=False):
        self.model = copy.deepcopy(model)#需要对MLP的模型进行改进，0代表模型，1代表模型名称
        #self.latent_model=copy.deepcopy(latent_model)
        #self.model_name = model[1]
        self.id = id  # integer
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        self.dim_latent=args.dim_latent
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.beta = args.beta
        self.lamda = args.lamda
        self.local_epochs = args.local_epochs
        self.algorithm = args.algorithm
        self.K = args.K
        self.dataset = args.dataset
        self.train_data = train_data
        self.test_data = test_data
        #self.trainloader = DataLoader(train_data, self.batch_size, drop_last=False)
        self.trainloader = DataLoader(train_data, self.batch_size, shuffle=True, drop_last=True)
        self.testloader =  DataLoader(test_data, self.batch_size, drop_last=False)
        self.testloaderfull = DataLoader(test_data, self.test_samples)
        self.trainloaderfull = DataLoader(train_data, self.train_samples)
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)
        dataset_name = get_dataset_name(self.dataset)
        #self.unique_labels = RUNCONFIGS[dataset_name]['unique_labels']#回归不存在这个
        self.generative_alpha = RUNCONFIGS[dataset_name]['generative_alpha']
        self.generative_beta = RUNCONFIGS[dataset_name]['generative_beta']

        # those parameters are for personalized federated learning.
        self.local_model = copy.deepcopy(list(self.model.parameters()))
        self.personalized_model_bar = copy.deepcopy(list(self.model.parameters()))
        self.prior_decoder = None
        self.prior_params = None

        self.init_loss_fn()
        if use_adam:
            self.optimizer=torch.optim.Adam(
                params=self.model.parameters(),
                lr=self.learning_rate, betas=(0.9, 0.999),
                eps=1e-08, weight_decay=1e-2, amsgrad=False)
        else:
            self.optimizer = pFedIBOptimizer(self.model.parameters(), lr=self.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.99)
        self.label_counts = {}#改




    def init_loss_fn(self):#修改损失函数
        self.loss=nn.MSELoss()
       # self.dist_loss = nn.MSELoss()
        self.ensemble_loss=RKLDivLoss()
       # self.ce_loss = nn.CrossEntropyLoss()

    def set_parameters(self, model,beta=1):
        for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(), self.local_model):
            if beta == 1:
                old_param.data = new_param.data.clone()
                local_param.data = new_param.data.clone()
            else:
                old_param.data = beta * new_param.data.clone() + (1 - beta)  * old_param.data.clone()
                local_param.data = beta * new_param.data.clone() + (1-beta) * local_param.data.clone()

    def set_prior_decoder(self, model, beta=1):
        for new_param, local_param in zip(model.personal_layers, self.prior_decoder):
            if beta == 1:
                local_param.data = new_param.data.clone()
            else:
                local_param.data = beta * new_param.data.clone() + (1 - beta) * local_param.data.clone()


    def set_prior(self, model):
        for new_param, local_param in zip(model.get_encoder() + model.get_decoder(), self.prior_params):
            local_param.data = new_param.data.clone()

    # only for pFedMAS
    def set_mask(self, mask_model):
        for new_param, local_param in zip(mask_model.get_masks(), self.mask_model.get_masks()):
            local_param.data = new_param.data.clone()

    def set_shared_parameters(self, model, mode='decode'):
        # only copy shared parameters to local
        for old_param, new_param in zip(
                self.model.get_parameters_by_keyword(mode),
                model.get_parameters_by_keyword(mode)
        ):
            old_param.data = new_param.data.clone()

    def get_parameters(self):
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()


    def clone_model_paramenter(self, param, clone_param):
        with torch.no_grad():
            for param, clone_param in zip(param, clone_param):
                clone_param.data = param.data.clone()
        return clone_param
    
    def get_updated_parameters(self):
        return self.local_weight_updated
    
    def update_parameters(self, new_params, keyword='all'):
        for param , new_param in zip(self.model.parameters(), new_params):
            param.data = new_param.data.clone()

    def get_grads(self):
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad.data)
        return grads

   # def test(self):
    #    self.model.eval()
    #    test_acc = 0
    #    loss = 0
    #    for x, y in self.testloaderfull:
    #        output = self.model(x)['output']
    #        loss += self.loss(output, y)#修改
    #        test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()#修改
    #    return test_acc, loss, y.shape[0]
    
    def test(self,net_glob=None,net_preproc=None,n=0):
        self.model.eval()
        total_loss = 0
        total_mse = 0  # 用于存储累计的 MSE
        num_samples = 0
        for x, y in self.testloaderfull:
            x = x.float()#新加入的修改bug
            if net_preproc is not None:
                pca = PCA(n_components=n)
                X_pcam = pca.fit_transform(x)
                X_pcam = torch.tensor(X_pcam, dtype=torch.float32)
                X_pca = net_preproc(X_pcam)
                output = net_glob(X_pca)['output'].squeeze(-1)
            else:
               pca = PCA(n_components=n)
               X_pca = pca.fit_transform(x)
               X_pca = torch.tensor(X_pca, dtype=torch.float32)
               output = self.model(X_pca)['output'].squeeze(-1)
            loss = self.loss(output, y)  # 计算损失
            total_loss += loss.item() * y.size(0)  # 将损失乘以样本数量，方便后续计算平均损失

            # 计算 MSE 并累加
            mse = torch.mean((output - y) ** 2).item()  # 计算当前批次的 MSE
            total_mse += mse * y.size(0)  # 累加每个样本的 MSE

            num_samples += y.size(0)  # 统计样本总数

        average_loss = total_loss / num_samples  # 计算平均损失
        average_mse = total_mse / num_samples  # 计算平均 MSE
        return average_mse, average_loss, num_samples




    def test_personalized_model(self,net_glob=None,net_preproc=None,n=0):
        self.model.eval()
        test_acc = 0
        loss = 0
        test_mse=0
        self.update_parameters(self.personalized_model_bar)
        for x, y in self.testloaderfull:
            if net_preproc is not None:
                pca = PCA(n_components=n)
                X_pcam = pca.fit_transform(x)
                X_pcam = torch.tensor(X_pcam, dtype=torch.float32)
                X_pca = net_preproc(X_pcam)
                output = net_glob(X_pca)['output'].squeeze(-1)
            else:
               pca = PCA(n_components=n)
               X_pca = pca.fit_transform(x)
               X_pca = torch.tensor(X_pca, dtype=torch.float32)
               output = self.model(X_pca)['output'].squeeze(-1)
            loss += self.loss(output, y)
            #test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            test_mse += torch.mean((output - y) ** 2).item()
            #@loss += self.loss(output, y)
            #print(self.id + ", Test Accuracy:", test_acc / y.shape[0] )
            #print(self.id + ", Test Loss:", loss)
        self.update_parameters(self.local_model)
        return test_mse, y.shape[0], loss


    def get_next_train_batch(self,maptrainloader,iter_maptrainloader):#count_labels要给出false
        try:
            # Samples a new batch for personalizing
            (X, y) = next(iter_maptrainloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            iter_maptrainloader = iter(maptrainloader)
            (X, y) = next(iter_maptrainloader)
        result = {'X': X, 'y': y}
        return result

    def get_next_test_batch(self):
        try:
            # Samples a new batch for personalizing
            (X, y) = next(self.iter_testloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_testloader = iter(self.testloader)
            (X, y) = next(self.iter_testloader)
        return (X, y)

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "user_" + self.id + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        self.model = torch.load(os.path.join(model_path, "server" + ".pt"))
    
    @staticmethod
    def model_exists():
        return os.path.exists(os.path.join("models", "server" + ".pt"))
