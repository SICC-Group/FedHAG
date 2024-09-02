import torch
import torch.nn.functional as F
import numpy as np
from FLAlgorithms.users.userbase import User
from utils.model_utils import create_model

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

class UserFedHAG(User):
    def __init__(self,
                 args, id, model, generative_model,
                 train_data, test_data, latent_layer_idx,
                 use_adam=False):
        super().__init__(args, id, model, train_data, test_data, use_adam=use_adam)
        self.gen_batch_size = args.gen_batch_size
        self.generative_model = generative_model
        #self.latent_model=latent_model
        self.latent_layer_idx = latent_layer_idx
        self.num_features = 1
        self.RegressionTracker = RegressionTracker(self.num_features)
        #self.available_labels = available_labels
        #self.label_info=label_info


    def exp_lr_scheduler(self, epoch, decay=0.98, init_lr=0.1, lr_decay_epoch=1):
        """Decay learning rate by a factor of 0.95 every lr_decay_epoch epochs."""
        lr= max(1e-4, init_lr * (decay ** (epoch // lr_decay_epoch)))
        return lr

    def update_label_counts(self, labels, counts):
        for label, count in zip(labels, counts):
            self.label_counts[int(label)] += count

    def clean_up_counts(self):
        del self.label_counts
        self.label_counts = {label:1 for label in range(self.unique_labels)}

    def train(self, net,maptrainloader,iter_maptrainloader,glob_iter, global_mean, global_variance, prior,personalized=False, early_stop=100, regularization=True, verbose=False):
        #self.clean_up_counts()
        net.train()
        self.generative_model.eval()
        #self.latent_model.eval()
        TEACHER_LOSS, DIST_LOSS, LATENT_LOSS = 0, 0, 0
        RegressionTracker0 = RegressionTracker(self.num_features)
        RegressionTracker1= RegressionTracker(self.num_features)
        self.optimizer = torch.optim.Adam(net.parameters(), lr=0.002)
        #self.optimizer1 = torch.optim.Adam(self.model.parameters(), lr=0.002)
        for epoch in range(self.local_epochs):
            net.train()
            for i in range(self.K):
                self.optimizer.zero_grad()
                #### sample from real dataset (un-weighted)
                samples =self.get_next_train_batch(maptrainloader,iter_maptrainloader)
                X, y = samples['X'], samples['y']
                #f_num=X.shape[1]
                y=y.float()
                y = y.unsqueeze(-1)
                #self.update_label_counts(samples['labels'], samples['counts'])
                model_result=net(X.float())
                user_output_logp = model_result['output']
                RegressionTracker0.update(user_output_logp.detach())
                self.RegressionTracker.update(user_output_logp.detach())
                predictive_loss=self.loss(user_output_logp, y)
                mean, variance = RegressionTracker0.avg_and_var()
                

                #### sample y and generate z
                if regularization and epoch < early_stop:
                    #self.optimizer1.zero_grad()
                    #self.latent_model.train()
                    generative_alpha=self.exp_lr_scheduler(glob_iter, decay=0.98, init_lr=self.generative_alpha)
                    generative_beta=self.exp_lr_scheduler(glob_iter, decay=0.98, init_lr=self.generative_beta)
                    ### get generator output(latent representation) of the same label
                    gen_output=self.generative_model(y, latent_layer_idx=self.latent_layer_idx,prior=prior)['output']
                    logit_given_gen=net(gen_output,start_layer_idx=1)['output']
                    #target_p=F.softmax(logit_given_gen, dim=1).clone().detach()
                    RegressionTracker1.update(logit_given_gen.detach())
                    mean1, variance1 = RegressionTracker1.avg_and_var()
                    #user_latent_loss= generative_beta * self.ensemble_loss(mean, variance, mean1, variance1)
                    user_latent_loss= self.ensemble_loss(mean, variance, mean1, variance1)

                    reg_loss = self.ensemble_loss(global_mean, global_variance, mean, variance)

                    sampled_y=torch.randn(32) * global_variance.sqrt() + global_mean
                    sampled_y=sampled_y.clone().detach()
                    gen_result=self.generative_model(sampled_y, latent_layer_idx=self.latent_layer_idx,prior=prior)
                    gen_output=gen_result['output'] # latent representation when latent = True, x otherwise
                    user_output_logp =net(gen_output,start_layer_idx=1)['output']
                    # teacher_loss =  generative_alpha * torch.mean(
                    #     self.generative_model.dist_loss(user_output_logp, sampled_y.unsqueeze(-1))
                    # )
                    teacher_loss = torch.mean(self.generative_model.dist_loss(user_output_logp, sampled_y.unsqueeze(-1)))
                    # this is to further balance oversampled down-sampled synthetic data
                    gen_ratio = self.gen_batch_size / self.batch_size
                    loss=0.5*predictive_loss + 0.2*reg_loss+0.2* teacher_loss + 0.1*user_latent_loss
                    TEACHER_LOSS+=teacher_loss
                    LATENT_LOSS+=user_latent_loss
                else:
                    #### get loss and perform optimization
                    loss=predictive_loss
                loss.backward()
                self.optimizer.step()#self.local_model)
                #if regularization and epoch < early_stop:
                   #self.optimizer1.step()
        # local-model <=== self.model
        self.clone_model_paramenter(net.parameters(), self.local_model)
        if personalized:
            self.clone_model_paramenter(net.parameters(), self.personalized_model_bar)
        #self.lr_scheduler.step(glob_iter)
        if regularization and verbose:
            TEACHER_LOSS=TEACHER_LOSS.detach().numpy() / (self.local_epochs * self.K)
            LATENT_LOSS=LATENT_LOSS.detach().numpy() / (self.local_epochs * self.K)
            info='\nUser Teacher Loss={:.4f}'.format(TEACHER_LOSS)
            info+=', Latent Loss={:.4f}'.format(LATENT_LOSS)
            print(info)
        return net.state_dict()

    def adjust_weights(self, samples):
        labels, counts = samples['labels'], samples['counts']
        #weight=self.label_weights[y][:, user_idx].reshape(-1, 1)
        np_y = samples['y'].detach().numpy()
        n_labels = samples['y'].shape[0]
        weights = np.array([n_labels / count for count in counts]) # smaller count --> larger weight
        weights = len(self.available_labels) * weights / np.sum(weights) # normalized
        label_weights = np.ones(self.unique_labels)
        label_weights[labels] = weights
        sample_weights = label_weights[np_y]
        return sample_weights


