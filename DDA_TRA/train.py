import sys
import torch.optim as optim
import torch.utils.data
import numpy as np
from DDA_TRA.test import test
from DDA_TRA.model import CNNRNNModel
import os


def DANN_train(dataloader_source, dataloader_target, cuda, lr, n_epoch, num_class, kernel_size, second_dim, model_root):
    # load model
    my_net = CNNRNNModel(num_class=num_class, kernel_size=kernel_size, second_dim=second_dim)
    # setup optimizer
    optimizer = optim.Adam(my_net.parameters(), lr=lr)

    loss_class = torch.nn.NLLLoss()
    loss_domain = torch.nn.NLLLoss()

    if cuda:
        my_net = my_net.cuda()
        loss_class = loss_class.cuda()
        loss_domain = loss_domain.cuda()

    for p in my_net.parameters():
        p.requires_grad = True

    # training
    best_accu_t = 0.0
    for epoch in range(n_epoch):

        len_dataloader = min(len(dataloader_source), len(dataloader_target))

        for i in range(len_dataloader):

            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # training model using source data
            for data_source in dataloader_source:
                # data_source = data_source_iter
                s_img, s_label = data_source

                my_net.zero_grad()
                batch_size = len(s_label)

                domain_label = torch.zeros(batch_size).long()

                if cuda:
                    s_img = s_img.cuda()
                    s_label = s_label.cuda()
                    domain_label = domain_label.cuda()

                class_output, domain_output = my_net(input_data=s_img, alpha=alpha)
                err_s_label = loss_class(class_output, s_label)
                err_s_domain = loss_domain(domain_output, domain_label)

                # training model using target data
                for data_target in dataloader_target:
                    t_img, _ = data_target

                    batch_size = len(t_img)

                    domain_label = torch.ones(batch_size).long()

                    if cuda:
                        t_img = t_img.cuda()
                        domain_label = domain_label.cuda()

                    _, domain_output = my_net(input_data=t_img, alpha=alpha)
                    err_t_domain = loss_domain(domain_output, domain_label)
                    err = err_t_domain + err_s_domain + err_s_label
                    err.backward()
                    optimizer.step()

                    #sys.stdout.write(
                    #    '\r epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
                    #    % (epoch, i + 1, len_dataloader, err_s_label.data.cpu().numpy(),
                    #       err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().item()))
                    #sys.stdout.flush()
                    if not os.path.exists(model_root):
                        os.makedirs(model_root)
                    torch.save(my_net, '{0}/model_epoch_current.pth'.format(model_root))

        #print('\n')
        accu_s = test(dataloader_source, cuda, model_root, alpha=alpha)
        #print('Accuracy of the %s dataset: %f' % ('source', accu_s))
        accu_t = test(dataloader_target, cuda, model_root, alpha=alpha)
        #print('Accuracy of the %s dataset: %f\n' % ('target', accu_t))
        if accu_t > best_accu_t:
            best_accu_s = accu_s
            best_accu_t = accu_t
            torch.save(my_net, '{0}/model_epoch_best.pth'.format(model_root))

    print('============ Summary ============= \n')
    print('Accuracy of the %s dataset: %f' % ('source', best_accu_s))
    print('Accuracy of the %s dataset: %f' % ('target', best_accu_t))
    print('Corresponding model was save in ' + model_root + '/model_epoch_best.pth')
