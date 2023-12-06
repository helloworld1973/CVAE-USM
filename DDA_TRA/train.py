import sys
import torch.optim as optim
import torch.utils.data
import numpy as np

from DDA_TRA import modelopera
from DDA_TRA.opt import get_optimizer
from DDA_TRA.test import test
from DDA_TRA.model import CNNRNNModel
import os

from DDA_TRA.util import set_random_seed, log_and_print, print_row


def DDA_TRA_train(S_torch_loader, T_torch_loader, ST_torch_loader, global_epoch, local_epoch_common, local_epoch_RNN,
                  local_epoch_temporal, time_lag_value,
                  conv1_in_channels, conv1_out_channels, conv2_out_channels,
                  full_connect_num, num_class, kernel_size, second_dim, GRL_alpha,
                  lr_decay, lr, optim_Adam_weight_decay, optim_Adam_beta, file_name, device):
    set_random_seed(1234)

    best_target_cm, best_target_acc, corresponding_best_source_acc = 0, 0, 0

    algorithm = CNNRNNModel(conv1_in_channels, conv1_out_channels, conv2_out_channels,
                            full_connect_num, num_class, kernel_size, second_dim)
    algorithm = algorithm.to(device)
    algorithm.train()
    opt_rnn_S = get_optimizer(algorithm, lr_decay, lr, optim_Adam_weight_decay, optim_Adam_beta,
                              nettype='DDA_TRA_rnn_S')
    opt_rnn_T = get_optimizer(algorithm, lr_decay, lr, optim_Adam_weight_decay, optim_Adam_beta,
                              nettype='DDA_TRA_rnn_T')
    opt_rnn_Temporal_Relation = get_optimizer(algorithm, lr_decay, lr, optim_Adam_weight_decay, optim_Adam_beta,
                                              nettype='DDA_TRA_Temporal_Relation')
    optc = get_optimizer(algorithm, lr_decay, lr, optim_Adam_weight_decay, optim_Adam_beta,
                         nettype='DDA_TRA_common')

    for round in range(global_epoch):
        print(f'\n========ROUND {round}========')
        log_and_print(f'\n========ROUND {round}========', filename=file_name)

        print('====RNN network training====')
        log_and_print('====RNN network training====', filename=file_name)
        loss_list = ['S_sequence_loss', 'T_sequence_loss']
        print_row(['epoch'] + [item + '_loss' for item in loss_list], colwidth=15, file_name=file_name)
        for step in range(local_epoch_RNN):
            for S_data in S_torch_loader:
                for T_data in T_torch_loader:
                    loss_result_dict = algorithm.forward_update_RNN_network(S_data, T_data, opt_rnn_S, opt_rnn_T,
                                                                            time_lags=time_lag_value)
            print_row([step] + [loss_result_dict[item] for item in loss_list], colwidth=15, file_name=file_name)

        print('====Temporal relation alignment====')
        log_and_print('====Temporal relation alignment====', filename=file_name)
        loss_list = ['temporal_loss', 'S_using_Trnn_temporal_loss', 'T_using_Srnn_temporal_loss']
        print_row(['epoch'] + [item + '_loss' for item in loss_list], colwidth=15, file_name=file_name)
        for step in range(local_epoch_temporal):
            for S_data in S_torch_loader:
                for T_data in T_torch_loader:
                    loss_result_dict = algorithm.forward_update_temporal_alignment(S_data, T_data,
                                                                                   opt_rnn_Temporal_Relation,
                                                                                   time_lag_value)
            print_row([step] + [loss_result_dict[item] for item in loss_list], colwidth=15, file_name=file_name)

        print('====DANN feature update====')
        log_and_print('====DANN feature update====', filename=file_name)
        loss_list = ['total', 'classes', 'domains']
        print_row(['epoch'] + [item + '_loss' for item in loss_list], colwidth=15, file_name=file_name)
        for step in range(local_epoch_common):
            for data in ST_torch_loader:
                for S_data in S_torch_loader:
                    loss_result_dict = algorithm.forward_update_common_components(data, S_data, optc, GRL_alpha)
            print_row([step] + [loss_result_dict[item] for item in loss_list], colwidth=15, file_name=file_name)

        print('====Evaluation====')
        log_and_print('====Evaluation====', filename=file_name)
        S_acc = modelopera.accuracy(algorithm, S_torch_loader, None)
        T_acc, T_cm = modelopera.accuracy_cm(algorithm, T_torch_loader, None)
        log_and_print('====S_acc====' + str(S_acc), filename=file_name)
        log_and_print('====T_acc====' + str(T_acc), filename=file_name)

        if T_acc > best_target_acc:
            best_target_acc = T_acc
            best_target_cm = T_cm
            corresponding_best_source_acc = S_acc

    return best_target_acc, best_target_cm, corresponding_best_source_acc
