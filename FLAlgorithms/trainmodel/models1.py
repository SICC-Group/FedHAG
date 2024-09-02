import torch.nn as nn
import torch.nn.functional as F
from utils.model_config import CONFIGS_

import collections

#################################
##### Neural Network model #####
#################################

class Net1(nn.Module):
    def __init__(self, dataset='material', model='cnn'):
        super(Net1, self).__init__()
        # define network layers
        print("Creating model for {}".format(dataset))
        self.dataset = dataset
        configs, input_channel, self.output_dim, self.hidden_dim, self.latent_dim = CONFIGS_[dataset]
        print('Network configs:', configs)
        self.named_layers, self.layers, self.layer_names = self.build_network(
            configs, input_channel, self.output_dim)
        self.n_parameters = len(list(self.parameters()))
        self.n_share_parameters = len(self.get_encoder())

    def get_number_of_parameters(self):
        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return pytorch_total_params

    def build_network(self, configs, input_channel, output_dim):
        layers = nn.ModuleList()
        named_layers = {}
        layer_names = []
        kernel_size, stride, padding = 3, 2, 1
        for i, x in enumerate(configs):
            if x == 'F':
                layer_name = 'flatten{}'.format(i)
                layer = nn.Flatten(1)
                layers += [layer]
                layer_names += [layer_name]
            elif x == 'M':
                pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
                layer_name = 'pool{}'.format(i)
                layers += [pool_layer]
                layer_names += [layer_name]
            else:
                cnn_name = 'encode_cnn{}'.format(i)
                cnn_layer = nn.Conv2d(input_channel, x, stride=stride, kernel_size=kernel_size, padding=padding)
                named_layers[cnn_name] = [cnn_layer.weight, cnn_layer.bias]

                bn_name = 'encode_batchnorm{}'.format(i)
                bn_layer = nn.BatchNorm2d(x)
                named_layers[bn_name] = [bn_layer.weight, bn_layer.bias]

                relu_name = 'relu{}'.format(i)
                relu_layer = nn.ReLU(inplace=True)  # no parameters to learn

                layers += [cnn_layer, bn_layer, relu_layer]
                layer_names += [cnn_name, bn_name, relu_name]
                input_channel = x

        # finally, regression layer
        fc_layer_name1 = 'encode_fc1'
        fc_layer1 = nn.Linear(self.hidden_dim, self.latent_dim)
        layers += [fc_layer1]
        layer_names += [fc_layer_name1]
        named_layers[fc_layer_name1] = [fc_layer1.weight, fc_layer1.bias]

        fc_layer_name = 'decode_fc2'
        fc_layer = nn.Linear(self.latent_dim, self.output_dim)
        layers += [fc_layer]
        layer_names += [fc_layer_name]
        named_layers[fc_layer_name] = [fc_layer.weight, fc_layer.bias]
        return named_layers, layers, layer_names

    def get_parameters_by_keyword(self, keyword='encode'):
        params = []
        for name, layer in zip(self.layer_names, self.layers):
            if keyword in name:
                params += [layer.weight, layer.bias]
        return params

    def get_encoder(self):
        return self.get_parameters_by_keyword("encode")

    def get_decoder(self):
        return self.get_parameters_by_keyword("decode")

    def get_shared_parameters(self, detach=False):
        return self.get_parameters_by_keyword("decode_fc2")

    def get_learnable_params(self):
        return self.get_encoder() + self.get_decoder()

    def forward(self, x, start_layer_idx=0, logit=False):  # 修改forward
        """
        :param x:
        :param logit: return logit vector before the last softmax layer (not used for regression)
        :param start_layer_idx: if 0, conduct normal forward; otherwise, forward from the last few layers (see mapping function)
        :return:
        """
        if start_layer_idx < 0:
            return self.mapping(x, start_layer_idx=start_layer_idx, logit=logit)
        
        results = {}
        z = x
        for idx in range(start_layer_idx, len(self.layers)):
            layer_name = self.layer_names[idx]
            layer = self.layers[idx]
            z = layer(z)

        # For regression, directly output the prediction
        results['output'] = z

        # Optionally return the logit (which in this case is the same as output)
        if logit:
            results['logit'] = z

        return results

    def mapping(self, z_input, start_layer_idx=-1, logit=True):
        z = z_input
        n_layers = len(self.layers)
        for layer_idx in range(n_layers + start_layer_idx, n_layers):
            layer = self.layers[layer_idx]
            z = layer(z)
        
        result = {'output': z}
        if logit:
            result['logit'] = z
        return result
    
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x, start_layer_idx=0):
        x = x.float()  # 将 x 转换为 float 类型

        if start_layer_idx <= 0:
            x = self.fc1(x)
            x = self.relu(x)
        if start_layer_idx <= 1:
            x = self.fc2(x)
        
        return x

class Net2(nn.Module):
    def __init__(self, input_size):
        super(Net2, self).__init__()
        self.mlp = MLP(input_size)
        self.n_parameters = len(list(self.parameters()))

    def forward(self, x, start_layer_idx=0):
        # Forward pass through the MLP from the specified layer
        results = {}
        z = self.mlp(x, start_layer_idx=start_layer_idx)
        results['output'] = z
        return results

    def get_number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

