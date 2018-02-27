import torch
from torch import Tensor
import dlc_practical_prologue as prologue


# EE–559: Practical Session 2


# 1 Nearest neighbor

def nearest_classification(train_input, train_target, x):
    distances = torch.sum(torch.pow(train_input - x, 2), 1)
    _ , indices = distances.sort()
    return train_target[indices[1]]


# 2 Error estimation

def compute_nb_errors(train_input, train_target, test_input, test_target, mean = None, proj = None):
    if mean is not None:
        train_input = train_input - mean.view(mean.size()[0], 1).expand_as(train_input)
        test_input  = test_input  - mean.view(mean.size()[0], 1).expand_as(test_input)

    #if proj is not None:

    return 0


# 3 PCA

def PCA(x):
    mean_x = 1/x.size()[0] * torch.sum(x, 1)
    eigen_values = torch.mm(x.transpose(1, 0), x).eig()


N, D, C = 10, 4, 2
train_input = torch.Tensor(N, D).normal_()


# 4 Check that all this makes sense
