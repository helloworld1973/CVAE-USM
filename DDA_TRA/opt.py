import torch


def get_params(alg, lr_decay, init_lr, nettype):
    if nettype == 'DDA_TRA_common':
        params = [
            {'params': alg.feature.parameters(), 'lr': lr_decay * init_lr},
            {'params': alg.class_classifier.parameters(), 'lr': lr_decay * init_lr},
            {'params': alg.domain_classifier.parameters(), 'lr': lr_decay * init_lr}

        ]
        return params
    elif nettype == 'DDA_TRA_rnn_S':
        params = [
            {'params': alg.source_rnn.parameters(), 'lr': lr_decay * init_lr}
        ]
        return params
    elif nettype == 'DDA_TRA_rnn_T':
        params = [
            {'params': alg.target_rnn.parameters(), 'lr': lr_decay * init_lr}
        ]
        return params
    elif nettype == 'DDA_TRA_Temporal_Relation':
        params = [
            {'params': alg.feature.parameters(), 'lr': lr_decay * init_lr}
        ]
        return params


def get_optimizer(alg, lr_decay, lr, optim_Adam_weight_decay, optim_Adam_beta, nettype):
    params = get_params(alg, lr_decay, lr, nettype=nettype)
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=optim_Adam_weight_decay, betas=(optim_Adam_beta, 0.9))
    return optimizer
