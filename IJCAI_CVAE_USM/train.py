import time
from IJCAI_CVAE_USM.alg.model import CVAE_USM
from IJCAI_CVAE_USM.alg.opt import *
from IJCAI_CVAE_USM.alg import modelopera
from IJCAI_CVAE_USM.utils.util import set_random_seed, log_and_print, print_row


def GPU_CVAE_USM_train(S_torch_loader, T_torch_loader, ST_torch_loader, global_epoch,
                       local_epoch, num_classes,
                       num_temporal_states,
                       conv1_in_channels, conv1_out_channels, conv2_out_channels,
                       kernel_size_num, in_features_size, hidden_size, dis_hidden,
                       ReverseLayer_latent_domain_alpha, variance, alpha, beta, gamma, delta,
                       epsilon,
                       lr_decay, lr, optim_Adam_weight_decay, optim_Adam_beta,
                       file_name, device):

    set_random_seed(1234)

    best_valid_acc, target_acc = 0, 0

    algorithm = CVAE_USM(conv1_in_channels, conv1_out_channels, conv2_out_channels, kernel_size_num,
                         in_features_size,
                         hidden_size, dis_hidden, num_classes, num_temporal_states,
                         ReverseLayer_latent_domain_alpha, variance,
                         alpha, beta, gamma, delta, epsilon)
    algorithm = algorithm.to(device)
    algorithm.train()
    opt = get_optimizer(algorithm, lr_decay, lr, optim_Adam_weight_decay, optim_Adam_beta,
                        nettype='CVAE_USM')

    for round in range(global_epoch):
        print(f'\n========ROUND {round}========')
        log_and_print(f'\n========ROUND {round}========', filename=file_name)

        loss_list = ['total', 'reconstruct', 'KL', 'source_classes', 'disc_domains', 'temporal']
        eval_dict = {'train': ['train_in'], 'valid': ['valid_in'], 'target': ['target_out'],
                     'target_cluster': ['target_out']}
        print_key = ['epoch']
        print_key.extend([item + '_loss' for item in loss_list])
        print_key.extend([item + '_acc' for item in eval_dict.keys()])
        print_key.append('total_cost_time')
        print_row(print_key, colwidth=15, file_name=file_name)

        sss = time.time()
        for step in range(local_epoch):
            temporal_state_labels_S = 0
            temporal_state_labels_T = 0
            for ST_data in ST_torch_loader:
                for S_data in S_torch_loader:
                    for T_data in T_torch_loader:
                        step_vals, temporal_state_labels_S, temporal_state_labels_T = algorithm.update(ST_data, S_data, T_data, opt, device)

            algorithm.GPU_set_tlabel(S_torch_loader, T_torch_loader, temporal_state_labels_S, temporal_state_labels_T, device)
            results = {'epoch': step, }

            results['train_acc'] = modelopera.GPU_accuracy(
                algorithm, S_torch_loader, None)

            print('source_acc___________________________________________________________________________________')
            log_and_print(
                'source_acc___________________________________________________________________________________',
                filename=file_name)
            acc, S_mu, S_y = modelopera.GPU_accuracy(algorithm, S_torch_loader, None)
            results['valid_acc'] = acc

            print('target_acc_#################################################################################')
            log_and_print(
                'target_acc_#################################################################################',
                filename=file_name)
            acc, cluster_acc, cm, T_mu, T_y = modelopera.GPU_accuracy_target_user(algorithm, T_torch_loader,
                                                                                  S_torch_loader, None)
            results['target_acc'] = acc
            results['target_cluster_acc'] = cluster_acc
            results['cm'] = cm

            for key in loss_list:
                results[key + '_loss'] = step_vals[key]
            if results['target_cluster_acc'] > best_valid_acc:
                best_valid_acc = results['target_cluster_acc']
                target_acc = results['target_cluster_acc']
                best_cm = results['cm']
            results['total_cost_time'] = time.time() - sss
            print_row([results[key] for key in print_key], colwidth=15, file_name=file_name)

    print(f'Target acc: {target_acc:.4f}')
    print(best_cm)
    log_and_print(f'Target acc: {target_acc:.4f}', filename=file_name)

    return target_acc, best_cm
