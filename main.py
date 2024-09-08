#!/usr/bin/env python
import argparse
from FLAlgorithms.servers.serveravg import FedAvg
from FLAlgorithms.servers.serverFedProx import FedProx
from FLAlgorithms.servers.serverFedDistill import FedDistill
from FLAlgorithms.servers.serverpFedGen import FedGen
from FLAlgorithms.servers.serverpFedEnsemble import FedEnsemble
from FLAlgorithms.servers.serverFedRegress import FedRegress
from FLAlgorithms.servers.serverFedHAG import FedHAG
from utils.model_utils import create_model
from utils.plot_utils import *
import torch
from multiprocessing import Pool

def create_server_n_user(args, i):
    model = create_model(args.dim_latent)#改成MLP
    if ('FedAvg' in args.algorithm):
        server=FedAvg(args, model, i)
    elif 'FedGen' in args.algorithm:
        server=FedGen(args, model, i)
    elif ('FedProx' in args.algorithm):
        server = FedProx(args, model, i)
    elif ('FedHAG' in args.algorithm):
       server = FedHAG(args, model, i)
    #elif ('S' in args.algorithm):
        #server = FedEnsemble(args, model, i)
    elif ('FedRegress' in args.algorithm):
        server = FedRegress(args, model, i)
    #else:
        #print("Algorithm {} has not been implemented.".format(args.algorithm))
        #exit()
    return server


def run_job(args, i):
    torch.manual_seed(i)
    print("\n\n         [ Start training iteration {} ]           \n\n".format(i))
    # Generate model
    server = create_server_n_user(args, i)
    if args.train:
        server.train(args)
        server.test()

def main(args):
    for i in range(args.times):
        run_job(args, i)
    print("Finished training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Material-alpha0.1-ratio0.5")#改
    #parser.add_argument("--model", type=str, default="cnn")#改成MLP
    parser.add_argument("--train", type=int, default=1, choices=[0,1])
    parser.add_argument("--algorithm", type=str, default="pFedHAG")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--preproc_batch_size", type=int, default=128)
    parser.add_argument("--gen_batch_size", type=int, default=64, help='number of samples from generator')#这个也需要改
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Local learning rate")
    parser.add_argument("--personal_learning_rate", type=float, default=0.01, help="Personalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument("--ensemble_lr", type=float, default=1e-4, help="Ensemble learning rate.")
    parser.add_argument("--beta", type=float, default=1.0, help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    parser.add_argument("--lamda", type=int, default=1, help="Regularization term")
    parser.add_argument("--mix_lambda", type=float, default=0.1, help="Mix lambda for FedMXI baseline")
    parser.add_argument("--embedding", type=int, default=0, help="Use embedding layer in generator network")#在生成网络中使用嵌入层
    parser.add_argument("--num_glob_iters", type=int, default=100)
    parser.add_argument("--local_epochs", type=int, default=20)
    parser.add_argument("--num_users", type=int, default=3, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=1, help="Computation steps")
    parser.add_argument("--times", type=int, default=1, help="running time")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"], help="run device (cpu | cuda)")
    parser.add_argument("--result_path", type=str, default="results", help="directory path to save results")


    #parser.add_argument('--alg', type=str, default='FLic', help="Algorithm")#111111
    #parser.add_argument('--dataset', type=str, default='ABX3', help="choice of the dataset")

    #parser.add_argument('--num_users', type=int, default=3, help="number of users")
    #parser.add_argument("--batch_size", type=int, default=64)
    #parser.add_argument('--shard_per_user', type=int, default=2, help="number of classes per user")
    #parser.add_argument('--num_classes', type=int, default=10, help="number of classes")

    #parser.add_argument('--frac', type=float, default=1, help="the fraction of clients: C")

    #parser.add_argument('--epochs', type=int, default=100,help="rounds of training")#11111111
    parser.add_argument('--local_bs', type=int, default=100, help="local batch size")#111111111
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate for model")
    #parser.add_argument('--local_ep', type=int, default=10, help="number of local epoch")111111111
    parser.add_argument('--local_rep_ep', type=int, default=1, help="number of local epoch for representation among local_ep")
    parser.add_argument('--reg_w', type=float, default=0.001, help="regularization of W ")
    parser.add_argument('--reg_reg_prior', type=float, default=0.001, help="regularization of W ")


    parser.add_argument('--model_type', type=str, default='reg', help="choosing the global model, [classif, no-hlayers, 2-hlayers]")
    parser.add_argument('--n_hidden', type=int, default=64, help="number of units in hidden layers")
    parser.add_argument('--dim_latent', type=int, default=16, help="latent dimension")
    parser.add_argument('--align_epochs', type=int, default=100, help="number of epochs for alignment during pretraining")
    parser.add_argument('--align_epochs_altern', type=int, default=5, help="number of epochs for alignment during alternate")
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

    args = parser.parse_args()

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    print("Ensemble learing rate       : {}".format(args.ensemble_lr))
    print("Average Moving       : {}".format(args.beta))
    print("Subset of users      : {}".format(args.num_users))
    print("Number of global rounds       : {}".format(args.num_glob_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset       : {}".format(args.dataset))
#    print("Local Model       : {}".format(args.model))
    print("Device            : {}".format(args.device))
    print("=" * 80)
    main(args)
