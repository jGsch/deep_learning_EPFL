import torch
from torch import Tensor
import dlc_practical_prologue as prologue


# EE–559: Practical Session 2


# 1 Nearest neighbor

def nearest_classification(train_input, train_target, x):
    distances = torch.norm(train_input - x, 2, 1)
    _ , indices = distances.sort()
    return train_target[indices[0]]


# 2 Error estimation

def compute_nb_errors(train_input, train_target, test_input, test_target, mean = None, proj = None):
    if mean is not None:
        train_input = train_input - mean
        test_input  = test_input - mean

    if proj is not None:
        train_input = torch.mm(train_input, proj.transpose(1, 0))
        test_input  = torch.mm(test_input, proj.transpose(1, 0))

    error = 0
    for index, test in enumerate(test_input):
        if not nearest_classification(train_input, train_target, test) == test_target[index]:
            error += 1

    return error


# 3 PCA

def PCA(x):
    mean = x.mean(0)
    b = x - mean

    eig_values, eig_vectors = torch.mm(x.transpose(1, 0), x).eig(True)
    _ , order = eig_values[:, 0].abs().sort(0, True)
    eig_vectors = eig_vectors.t()[order]

    return mean, eig_vectors


# 4 Check that all this makes sense

for cifar in [False, True]:
    if cifar: print("Cifar dataset...")
    else: print("MNIST dataset...")

    train_input, train_target, test_input, test_target = prologue.load_data(cifar = None)

    basis = train_input.new(100, train_input.size(1)).normal_()
    nb_error = compute_nb_errors(train_input, train_target, test_input, test_target, None, basis)
    print("Number of error: " + str(nb_error/test_input.size(0)))

    mean, basis = PCA(train_input)
    train_input
    for dimension in [3, 10, 50, 100]:
        basis_pca = basis[0:dimension,:]
        nb_error = compute_nb_errors(train_input, train_target, test_input, test_target, mean, basis_pca)
        print("%_error: " + str(nb_error/test_input.size(0)) + " (Dimension: " + str(dimension) + ")")
