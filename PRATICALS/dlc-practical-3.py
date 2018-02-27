import torch
from torch import Tensor
import dlc_practical_prologue as prologue


# EE–559: Practical Session 3


# 1 Activation function

def sigma(x):
    exp = torch.exp(train_input)
    return (exp - 1/exp)/(exp + 1/exp)

def dsigma(x):
    exp = torch.exp(train_input)
    return 1 - torch.pow(exp - 1/exp, 2)/torch.pow(exp + 1/exp, 2)


# 2 Activation function

def loss(v, t):
    return torch.sum(torch.pow(v - t, 2), 1)

def dloss(v, t):
    return 0


x = torch.Tensor(5, 3).normal_()
y = torch.Tensor(5, 3).normal_()
loss(x,y)
