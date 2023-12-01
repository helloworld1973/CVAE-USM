import os
import torch.utils.data


def test(dataloader_target, cuda, model_root, alpha):
    my_net = torch.load(os.path.join(model_root, 'model_epoch_current.pth'))
    my_net = my_net.eval()

    if cuda:
        my_net = my_net.cuda()

    n_total = 0
    n_correct = 0

    for data_target in dataloader_target:

        # test model using target data
        t_img, t_label = data_target

        batch_size = len(t_label)

        if cuda:
            t_img = t_img.cuda()
            t_label = t_label.cuda()

        class_output, _ = my_net(input_data=t_img, alpha=alpha)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(t_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

    accu = n_correct.data.numpy() * 1.0 / n_total

    return accu
