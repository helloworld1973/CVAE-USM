import numpy as np
import torch
import torch.nn as nn
from DANN.functions import ReverseLayerF
import torch.nn.functional as F

from DDA_TRA import cca_core


class CNNRNNModel(nn.Module):

    def __init__(self, conv1_in_channels=6, conv1_out_channels=16, conv2_out_channels=32, full_connect_num=100,
                 num_class=4, kernel_size=9, second_dim=69):
        super(CNNRNNModel, self).__init__()
        self.conv1_in_channels = conv1_in_channels
        self.conv1_out_channels = conv1_out_channels
        self.conv2_out_channels = conv2_out_channels
        self.kernel_size = kernel_size
        self.full_connect_num = full_connect_num
        self.num_class = num_class
        self.second_dim = second_dim

        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(self.conv1_in_channels, self.conv1_out_channels,
                                                     kernel_size=(1, self.kernel_size)))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(self.conv1_out_channels))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_pool1', nn.MaxPool2d(kernel_size=(1, 2), stride=2))

        self.feature.add_module('f_conv2', nn.Conv2d(self.conv1_out_channels, self.conv2_out_channels,
                                                     kernel_size=(1, self.kernel_size)))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(self.conv2_out_channels))
        self.feature.add_module('f_relu2', nn.ReLU(True))
        self.feature.add_module('f_pool2', nn.MaxPool2d(kernel_size=(1, 2), stride=2))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1',
                                         nn.Linear(self.second_dim, self.full_connect_num))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(self.full_connect_num))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(self.full_connect_num, self.full_connect_num))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(self.full_connect_num))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(self.full_connect_num, self.num_class))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1',
                                          nn.Linear(self.second_dim, self.full_connect_num))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(self.full_connect_num))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(self.full_connect_num, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

        # source user temporal relation
        self.input_size = self.second_dim
        self.source_rnn = nn.Sequential()
        self.source_rnn.add_module('S_rnn_0',
                                   GRUWithLayerNorm(input_size=self.input_size, hidden_size=self.input_size * 4, layer_norm_size=self.input_size * 4))
        self.source_rnn.add_module('S_rnn_1',
                                   GRUWithLayerNorm(input_size=self.input_size * 4, hidden_size=self.input_size, layer_norm_size=self.input_size))

        # target user temporal relation
        self.target_rnn = nn.Sequential()
        self.target_rnn.add_module('T_rnn_0',
                                   GRUWithLayerNorm(input_size=self.input_size, hidden_size=self.input_size * 4, layer_norm_size=self.input_size * 4))
        self.target_rnn.add_module('T_rnn_1',
                                   GRUWithLayerNorm(input_size=self.input_size * 4, hidden_size=self.input_size, layer_norm_size=self.input_size))

    def forward_update_common_components(self, all_input, S_input, opt, alpha):
        all_x = all_input[0].float()
        all_c = all_input[1].long()
        all_d = all_input[2].long()


        S_x = S_input[0].float()
        S_c = S_input[1].long()

        feature = self.feature(all_x)
        feature = feature.view(-1, self.conv2_out_channels * feature.size(-1))
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        domain_output = self.domain_classifier(reverse_feature)
        d_classifier_loss = F.cross_entropy(domain_output, all_d)

        S_feature = self.feature(S_x)
        S_feature = S_feature.view(-1, self.conv2_out_channels * S_feature.size(-1))
        S_class_output = self.class_classifier(S_feature)
        S_classifier_loss = F.cross_entropy(S_class_output, S_c)

        loss = S_classifier_loss + d_classifier_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        return {'total': loss.item(), 'classes': S_classifier_loss.item(), 'domains': d_classifier_loss.item()}

    def forward_update_RNN_network(self, S_input, T_input, S_opt, T_opt, time_lags):
        S_x = S_input[0].float()
        T_x = T_input[0].float()

        S_feature_sequence = self.feature(S_x)
        S_feature_sequence = S_feature_sequence.view(-1, self.conv2_out_channels * S_feature_sequence.size(-1))
        S_input_sequence = S_feature_sequence[0:-time_lags, :]
        S_baseline_sequence = S_feature_sequence[time_lags:, :]
        source_rnn_output = self.source_rnn(S_input_sequence)
        S_sequence_loss = F.cross_entropy(source_rnn_output, S_baseline_sequence)
        S_opt.zero_grad()
        S_sequence_loss.backward()
        S_opt.step()

        T_feature_sequence = self.feature(T_x)
        T_feature_sequence = T_feature_sequence.view(-1, self.conv2_out_channels * T_feature_sequence.size(-1))
        T_input_sequence = T_feature_sequence[0:-time_lags, :]
        T_baseline_sequence = T_feature_sequence[time_lags:, :]
        target_rnn_output = self.target_rnn(T_input_sequence)
        T_sequence_loss = F.cross_entropy(target_rnn_output, T_baseline_sequence)
        T_opt.zero_grad()
        T_sequence_loss.backward()
        T_opt.step()

        return {'S_sequence_loss': S_sequence_loss.item(), 'T_sequence_loss': T_sequence_loss.item()}

    def forward_update_temporal_alignment(self, S_input, T_input, opt, time_lags):
        S_x = S_input[0].float()
        T_x = T_input[0].float()

        S_feature_sequence = self.feature(S_x)
        S_feature_sequence = S_feature_sequence.view(-1, self.conv2_out_channels * S_feature_sequence.size(-1))
        S_input_sequence = S_feature_sequence[0:-time_lags, :]
        S_baseline_sequence = S_feature_sequence[time_lags:, :]
        source_rnn_output = self.target_rnn(S_input_sequence)
        S_using_Trnn_temporal_loss = F.cross_entropy(source_rnn_output, S_baseline_sequence)

        T_feature_sequence = self.feature(T_x)
        T_feature_sequence = T_feature_sequence.view(-1, self.conv2_out_channels * T_feature_sequence.size(-1))
        T_input_sequence = T_feature_sequence[0:-time_lags, :]
        T_baseline_sequence = T_feature_sequence[time_lags:, :]
        target_rnn_output = self.source_rnn(T_input_sequence)
        T_using_Srnn_temporal_loss = F.cross_entropy(target_rnn_output, T_baseline_sequence)

        temporal_loss = S_using_Trnn_temporal_loss + T_using_Srnn_temporal_loss

        opt.zero_grad()
        temporal_loss.backward()
        opt.step()

        return {'temporal_loss': temporal_loss.item(), 'S_using_Trnn_temporal_loss': S_using_Trnn_temporal_loss.item(),
                'T_using_Srnn_temporal_loss': T_using_Srnn_temporal_loss.item()}

    def init_weight(self):
        # orthogonal initialization of recurrent weights
        for name, param in self.named_parameters():
            if 'weight' in name and 'gru' in name:
                nn.init.orthogonal_(param, gain=1)

    def model_similarity_compare(self, S_params, T_params):
        loss = 0
        num_layers = len(S_params)
        for i in range(num_layers):
            S_a_act_a_layer = S_params[i][1]
            S_a_act_a_layer = S_a_act_a_layer.detach().numpy()
            if len(S_a_act_a_layer.shape) == 1:
                S_a_act_a_layer = S_a_act_a_layer.reshape((S_a_act_a_layer.shape[0], 1))

            T_a_act_a_layer = T_params[i][1]
            T_a_act_a_layer = T_a_act_a_layer.detach().numpy()
            if len(T_a_act_a_layer.shape) == 1:
                T_a_act_a_layer = T_a_act_a_layer.reshape((T_a_act_a_layer.shape[0], 1))

            # CCA
            a_results = cca_core.get_cca_similarity(S_a_act_a_layer, T_a_act_a_layer, epsilon=1e-10, verbose=False)
            print("Mean CCA similarity" + 'Layer_' + str(i), np.mean(a_results["cca_coef1"]))
            loss += a_results
        return loss

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(-1, self.conv2_out_channels * feature.size(-1))
        return self.class_classifier(feature)

class GRUWithLayerNorm(nn.Module):
    def __init__(self, input_size, hidden_size, layer_norm_size, num_layers=1, batch_first=True, dropout=0, bias=False, eps=0):
        super(GRUWithLayerNorm, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=batch_first, dropout=dropout, bias=bias)
        self.layer_norm = nn.LayerNorm(layer_norm_size, eps=eps)

    def forward(self, x):
        output, _ = self.gru(x)
        return self.layer_norm(output)