import torch


def get_params(alg, lr_decay, init_lr, nettype):
    if nettype == 'CVAE_USM':
        params = [{'params': alg.CVAE_encoder.parameters(), 'lr': lr_decay * init_lr},
                  {'params': alg.CVAE_reparameterize.parameters(), 'lr': lr_decay * init_lr},
                  {'params': alg.CVAE_decoder.parameters(), 'lr': lr_decay * init_lr},
                  {'params': alg.classify_source.parameters(), 'lr': lr_decay * init_lr},
                  {'params': alg.domains.parameters(), 'lr': lr_decay * init_lr}]
        return params


def get_optimizer(alg, lr_decay, lr, optim_Adam_weight_decay, optim_Adam_beta, nettype):
    params = get_params(alg, lr_decay, lr, nettype=nettype)
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=optim_Adam_weight_decay, betas=(optim_Adam_beta, 0.9))
    return optimizer
