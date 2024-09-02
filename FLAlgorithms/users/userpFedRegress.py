import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from FLAlgorithms.users.userbase import User
#from FLAlgorithms.optimizers.fedoptimizer import pFedIBOptimizer
#from torch.nn.modules.loss import RKLDivLoss

class LogitTracker():
    def __init__(self, unique_labels):
        self.unique_labels = unique_labels
        self.labels = [i for i in range(unique_labels)]
        self.label_counts = torch.ones(unique_labels) # 记录每个标签的数据，这个得改
        self.logit_sums = torch.zeros((unique_labels,unique_labels) ) #用于存储每个标签的 logit 和，也需要改

    def update(self, logits, Y):
        """
        update logit tracker.
        :param logits: shape = n_sampls * logit-dimension
        :param Y: shape = n_samples
        :return: nothing
        """
        batch_unique_labels, batch_labels_counts = Y.unique(dim=0, return_counts=True)
        self.label_counts[batch_unique_labels] += batch_labels_counts
        # expand label dimension to be n_samples X logit_dimension
        labels = Y.view(Y.size(0), 1).expand(-1, logits.size(1))
        logit_sums_ = torch.zeros((self.unique_labels, self.unique_labels) )
        logit_sums_.scatter_add_(0, labels, logits)
        self.logit_sums += logit_sums_


    def avg(self):
        res= self.logit_sums / self.label_counts.float().unsqueeze(1)#改公式
        return res

class RegressionTracker():
    def __init__(self, num_features):
        """
        初始化存储预测值的和以及平方和的变量。
        :param num_features: 回归任务中预测值的维度（通常是1，如果是多输出回归，则大于1）。
        """
        self.num_features = num_features
        self.pred_sum = torch.zeros(num_features)  # 用于存储预测值的和
        self.pred_square_sum = torch.zeros(num_features)  # 用于存储预测值平方的和
        self.count = 0  # 记录样本数量

    def update(self, predictions):
        """
        更新回归预测值的和以及平方和。
        :param predictions: shape = n_samples * num_features
        """
        # 确保 pred_sum 和 pred_square_sum 在累加时不会被原地修改
        self.pred_sum = self.pred_sum + predictions.sum(dim=0)  # 修改累加操作
        self.pred_square_sum = self.pred_square_sum + (predictions ** 2).sum(dim=0)  # 修改累加操作
        self.count += predictions.size(0)  # 更新样本数量

    def avg_and_var(self):
        """
        计算预测值分布的均值和方差。
        :return: 预测值均值和方差的张量
        """
        if self.count == 0:
            raise ValueError("Count is zero, cannot compute mean and variance.")
        
        mean = self.pred_sum / self.count  # 计算均值
        variance = (self.pred_square_sum / self.count) - (mean ** 2)  # 计算方差
        
        # 确保返回的 mean 和 variance 的梯度计算没有问题
        return mean.detach(), variance.detach()



class UserFedRegress(User):
    """
    Track and average logit vectors for each label, and share it with server/other users.
    """
    def __init__(self, args, id, model, train_data, test_data, use_adam=False):
        super().__init__(args, id, model, train_data, test_data, use_adam=use_adam)

        self.init_loss_fn()
        #self.unique_labels = unique_labels#改
        self.num_features = 1
        self.label_counts = {}
        self.RegressionTracker = RegressionTracker(self.num_features)
        self.global_mean = None
        self.global_variance = None
        self.reg_alpha = 1

    def update_label_counts(self, labels, counts):
        for label, count in zip(labels, counts):
            self.label_counts[int(label)] += count

    def clean_up_counts(self):#这个需要修改
        del self.label_counts
        self.label_counts = {int(label):1 for label in range(self.unique_labels)}

    def train(self, glob_iter, personalized=True, lr_decay=True, verbose=True):
        #self.clean_up_counts()
        torch.autograd.set_detect_anomaly(True)
        self.model.train()
        REG_LOSS, TRAIN_LOSS = 0.0, 0.0
        self.mean=0.0
        self.variance=0.0
        #rkl_div_loss = RKLDivLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.002)
        for epoch in range(1, self.local_epochs + 1):
            self.model.train()
            for i in range(self.K):
                result =self.get_next_train_batch(self.trainloader,self.iter_trainloader)
                X, y = result['X'], result['y']
                #if count_labels:
                    #self.update_label_counts(result['labels'], result['counts'])#计算每个标签的数量，需要改
                self.optimizer.zero_grad()
                result=self.model(X.float())
                output= result['output']
                self.RegressionTracker.update(output)
                y=y.float()
                #self.logit_tracker.update(logit, y)
                if self.global_mean is not None:
                    ### get desired logit for each sample
                    train_loss = self.loss(output, y)
                    #target_p = F.softmax(self.global_logits[y,:], dim=1)
                    mean, variance = self.RegressionTracker.avg_and_var()
                    reg_loss = self.ensemble_loss(self.global_mean, self.global_variance, mean, variance)
                    REG_LOSS =REG_LOSS + reg_loss.item()  # 使用 .item() 将张量转换为 float
                    TRAIN_LOSS =TRAIN_LOSS + train_loss.item()
                    loss = 0.6*train_loss + 0.4 * reg_loss
                    #loss=train_loss
                    #REG_LOSS += reg_loss
                    #TRAIN_LOSS += train_loss
                    #loss = train_loss + self.reg_alpha * reg_loss
                else:

                    loss=self.loss(output, y)
                #loss.backward(retain_graph=True)
                loss.backward()
                self.optimizer.step()#self.local_model)
            # local-model <=== self.model
            self.clone_model_paramenter(self.model.parameters(), self.local_model)
            if personalized:
                self.clone_model_paramenter(self.model.parameters(), self.personalized_model_bar)
        if lr_decay:
            self.lr_scheduler.step(glob_iter)
        if self.global_mean is not None and verbose:  # 使用 is not None 而不是 != None
            REG_LOSS =REG_LOSS / (self.local_epochs * self.K)  # 修改点：移除 detach().numpy()，直接处理浮点数
            TRAIN_LOSS =TRAIN_LOSS / (self.local_epochs * self.K)
            info = "Train loss {:.2f}, Regularization loss {:.2f}".format(TRAIN_LOSS, REG_LOSS)
            #info = "Train loss {:.2f}".format(TRAIN_LOSS)
            print(info)



