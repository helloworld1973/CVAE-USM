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


def accuracy_cm(network, t_loader, weights, usedpredict='p'):
    correct = 0
    weights_offset = 0
    t_length = len(t_loader.dataset.tensors[1])

    # Get all unique classes from the target loader
    all_labels = torch.cat([t_data[1] for t_data in t_loader])
    class_unique_elements = torch.unique(all_labels).numpy()
    num_classes = len(class_unique_elements)

    # Initialize the confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes))

    network.eval()
    with torch.no_grad():
        for t_data in t_loader:
            t_x = t_data[0].float()
            t_y = t_data[1].long()
            t_y = t_y.cpu()
            if usedpredict == 'p':
                p = network.predict(t_x)
                p = p.cpu()
            else:
                p = network.predict1(t_x)
                p = p.cpu()

            if weights is None:
                batch_weights = torch.ones(len(t_x))
            else:
                batch_weights = weights[weights_offset:
                                        weights_offset + len(t_x)]
                weights_offset += len(t_x)

            # Predictions
            if p.size(1) == 1:
                predictions = p.gt(0).view(-1)
            else:
                predictions = p.argmax(1)

            # Update the confusion matrix
            for i in range(len(t_y)):
                true_idx = np.where(class_unique_elements == t_y[i].item())[0][0]
                pred_idx = np.where(class_unique_elements == predictions[i].item())[0][0]
                confusion_matrix[true_idx, pred_idx] += 1

            # Calculate correct predictions
            correct += (predictions.eq(t_y).float() * batch_weights).sum().item()

    # Calculate the accuracy
    accuracy = correct / t_length

    return accuracy, confusion_matrix

