import numpy as np
import torch
from torch import softmax

def accuracy(network, loader, weights, usedpredict='p'):
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for data in loader:
            x = data[0].float()
            y = data[1].long()
            y = y.cpu()
            if usedpredict == 'p':
                p = network.predict(x)
                p = p.cpu()
            else:
                p = network.predict1(x)
                p = p.cpu()
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset:
                                        weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() *
                            batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() *
                            batch_weights).sum().item()

            total += batch_weights.sum().item()
    network.train()

    return correct / total
