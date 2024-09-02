from FLAlgorithms.users.userpFedGen import UserpFedGen
#from FLAlgorithms.users.userpFedGen import RegressionTracker
from FLAlgorithms.servers.serverbase import Server
from FLAlgorithms.trainmodel.RKLDivLoss import RKLDivLoss
from FLAlgorithms.users.userpFedGen import RegressionTracker
from utils.model_utils import create_model
from utils.model_utils import read_data, read_user_data, aggregate_user_data, create_generative_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
import os
import copy
import time
MIN_SAMPLES_PER_LABEL=1

class FedGen(Server):
    def __init__(self, args, model, seed):
        super().__init__(args, model, seed)

        # Initialize data for all users
        data = read_data(args.dataset)
        # data contains: clients, groups, train_data, test_data, proxy_data
        clients = data[0]
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
        self.generative_model = create_generative_model(args.dataset, args.algorithm, args.embedding)#生成器模型
        if not args.train:
            print('number of generator parameteres: [{}]'.format(self.generative_model.get_number_of_parameters()))
            print('number of model parameteres: [{}]'.format(self.model.get_number_of_parameters()))
        self.latent_layer_idx = self.generative_model.latent_layer_idx
        self.init_ensemble_configs()
        print("latent_layer_idx: {}".format(self.latent_layer_idx))
        #print("label embedding {}".format(self.generative_model.embedding))
        print("ensemeble learning rate: {}".format(self.ensemble_lr))
        print("ensemeble alpha = {}, beta = {}, eta = {}".format(self.ensemble_alpha, self.ensemble_beta, self.ensemble_eta))
        print("generator alpha = {}, beta = {}".format(self.generative_alpha, self.generative_beta))
        self.init_loss_fn() #初始化损失函数
        self.train_data_loader, self.train_iter = aggregate_user_data(data, args.dataset, self.ensemble_batch_size) #聚合用户数据(但不应该对用户数据进行聚合呀？)
        self.generative_optimizer = torch.optim.Adam(
            params=self.generative_model.parameters(),
            lr=self.ensemble_lr, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=self.weight_decay, amsgrad=False)
        self.generative_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.generative_optimizer, gamma=0.98)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.ensemble_lr, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=0, amsgrad=False)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.98)

        #### creating users ####
        self.users = []
        for i in range(total_users):
            id, train_data, test_data=read_user_data(i, data, dataset=args.dataset,)
            self.total_train_samples+=len(train_data)
            self.total_test_samples += len(test_data)
            id, train, test=read_user_data(i, data, dataset=args.dataset)
            user=UserpFedGen(
                args, id, model, self.generative_model,
                train_data, test_data, self.latent_layer_idx,
                use_adam=self.use_adam)
            self.users.append(user)
        print("Number of Train/Test samples:", self.total_train_samples, self.total_test_samples)
        print("Data from {} users in total.".format(total_users))
        print("Finished creating FedAvg server.")

    def train(self, args):
        #### pretraining
        for glob_iter in range(self.num_glob_iters):
            print("\n\n-------------Round number: ",glob_iter, " -------------\n\n")
            self.selected_users=self.select_users(glob_iter, self.num_users, return_idx=True)#选择用户为10个
            if not self.local:
                self.send_parameters(mode=self.mode)# broadcast averaged prediction model
            self.evaluate() #输出的是average glocal accurancy，Loss
            self.send_logits()
            #chosen_verbose_user = np.random.randint(0, len(self.users))
            self.timestamp = time.time() # log user-training start time
            for user in self.selected_users: # allow selected users to train
                #verbose= user_id == chosen_verbose_user
                # perform regularization using generated samples after the first communication round
                user.train(
                    glob_iter, self.global_mean, self.global_variance,
                    personalized=self.personalized, 
                    early_stop=self.early_stop,
                    verbose=True and glob_iter > 0,
                    regularization= glob_iter > 0 )#计算本地模型的总损失，包括预测损失、教师损失和潜在损失。这一个过程只更新本地模型参数
            curr_timestamp = time.time() # log  user-training end time
            train_time = (curr_timestamp - self.timestamp) / len(self.selected_users)
            self.metrics['user_train_time'].append(train_time)
            self.aggregate_logits()
            if self.personalized:
                self.evaluate_personalized_model()

            self.timestamp = time.time() # log server-agg start time
            self.train_generator(
                self.batch_size, self.global_mean, self.global_variance,
                epoches=self.ensemble_epochs // self.n_teacher_iters,
                latent_layer_idx=self.latent_layer_idx,
                verbose=True
            ) #更新生成器模型，不更新本地模型参数。其中损失函数包括，教师损失（生成器的输出作为预测模型的输入，得到的预测标签与随机生成标签之间的差别），多样性损失（生成样本与噪声之间的差别）
            self.aggregate_parameters()
            curr_timestamp=time.time()  # log  server-agg end time
            agg_time = curr_timestamp - self.timestamp
            self.metrics['server_agg_time'].append(agg_time)
            #if glob_iter  > 0 and glob_iter % 20 == 0 and self.latent_layer_idx == 0:
                #self.visualize_images(self.generative_model, glob_iter, repeats=10)

        self.save_results(args)
        self.save_model()

    def train_generator(self, batch_size, global_mean, global_variance,epoches=1, latent_layer_idx=0, verbose=False):
        """
        Learn a generator that find a consensus latent representation z, given a label 'y'.
        :param batch_size:
        :param epoches:
        :param latent_layer_idx: if set to -1 (-2), get latent representation of the last (or 2nd to last) layer.
        :param verbose: print loss information.
        :return: Do not return anything.
        """
        #self.generative_model.train()
        #self.model.eval()
        #self.generative_regularizer.train()
        #self.label_weights, self.qualified_labels = self.get_label_weights()
        TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS, STUDENT_LOSS2 = 0, 0, 0, 0
        rkl=RKLDivLoss()

        def update_generator_(n_iters, global_mean, global_variance, student_model, TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS):
            num_features=1
            RegressionTracker2=RegressionTracker(num_features)
            RegressionTracker3=RegressionTracker(num_features)
            self.generative_model.train()
            student_model.eval()
            #self.latent_model.eval()
            for i in range(n_iters):
                self.generative_optimizer.zero_grad()
                y = torch.randn(batch_size) * global_variance.sqrt() + global_mean

                y_input=y
                ## feed to generator
                gen_result=self.generative_model(y_input, latent_layer_idx=latent_layer_idx, verbose=True)
                # get approximation of Z( latent) if latent set to True, X( raw image) otherwise
                gen_output, eps=gen_result['output'], gen_result['eps']
                ##### get losses ####
                # decoded = self.generative_regularizer(gen_output)
                # regularization_loss = beta * self.generative_model.dist_loss(decoded, eps) # map generated z back to eps
                diversity_loss=self.generative_model.diversity_loss(eps, gen_output)  # encourage different outputs

                ######### get teacher loss ############
                teacher_loss=0
                mean_total=0
                variance_total=0
                mean_avg=0
                variance_avg=0
                teacher_logit=0
                for user_idx, user in enumerate(self.selected_users):
                    #user.latent_model.eval()
                    user.model.eval()
                    #weight=self.label_weights[y][:, user_idx].reshape(-1, 1)
                    #expand_weight=np.tile(weight, (1, self.unique_labels))
                    #gen_output = gen_output.permute(1, 0)
                    user_result_given_gen=user.model(gen_output,start_layer_idx=1)['output']
                    RegressionTracker2.update(user_result_given_gen)
                    mean, variance = RegressionTracker2.avg_and_var()
                    mean_total+=mean
                    variance_total+=variance
                    #user_output_logp_=F.log_softmax(user_result_given_gen['logit'], dim=1)
                    teacher_loss_=torch.mean(self.generative_model.dist_loss(user_result_given_gen, y_input))
                    teacher_loss+=teacher_loss_
                    #teacher_logit+=user_result_given_gen['logit'] * torch.tensor(expand_weight, dtype=torch.float32)
                mean_avg=mean_total/len(self.selected_users)
                variance_avg=variance_total/len(self.selected_users)
                ######### get student loss ############
                student_output=student_model(gen_output,start_layer_idx=1)['output']
                RegressionTracker3.update(student_output)
                mean1, variance1 = RegressionTracker3.avg_and_var()
                student_loss=rkl(mean_avg, variance_avg, mean1, variance1)
                if self.ensemble_beta > 0:
                    loss=self.ensemble_alpha * teacher_loss - self.ensemble_beta * student_loss + self.ensemble_eta * diversity_loss
                else:
                    loss=self.ensemble_alpha * teacher_loss + self.ensemble_eta * diversity_loss
                loss.backward()
                self.generative_optimizer.step()
                TEACHER_LOSS += self.ensemble_alpha * teacher_loss#(torch.mean(TEACHER_LOSS.double())).item()
                STUDENT_LOSS += self.ensemble_beta * student_loss#(torch.mean(student_loss.double())).item()这一项一直没有用
                DIVERSITY_LOSS += self.ensemble_eta * diversity_loss#(torch.mean(diversity_loss.double())).item()
            return TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS

        for i in range(epoches):
            TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS=update_generator_(
                self.n_teacher_iters, global_mean, global_variance, self.model, TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS)

        TEACHER_LOSS = TEACHER_LOSS.detach().numpy() / (self.n_teacher_iters * epoches)
        STUDENT_LOSS = STUDENT_LOSS.detach().numpy() / (self.n_teacher_iters * epoches)
        DIVERSITY_LOSS = DIVERSITY_LOSS.detach().numpy() / (self.n_teacher_iters * epoches)
        info="Generator: Teacher Loss= {:.4f}, Student Loss= {:.4f}, Diversity Loss = {:.4f}, ". \
            format(TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS)
        if verbose:
            print(info)
        self.generative_lr_scheduler.step()
    
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
