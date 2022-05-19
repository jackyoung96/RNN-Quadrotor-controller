import math
import random

import numpy as np
import matplotlib.pyplot as plt

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig("gradient_flow.png")

def rot_matrix_similarity(mat1, mat2):
    """
    similarity btw two rotation matrix
    """
    mat1 = mat1.reshape((-1,3,3))
    mat2 = np.transpose(mat2.reshape((3,3)))
    result = np.zeros((mat1.shape[0],))
    for i in range(mat1.shape[0]):
        R = np.matmul(mat1[i], mat2)
        cos = (np.trace(R)-1)/2
        cos1 = np.clip(cos, -1, 1)
        result[i] = np.arccos(cos1)
    return result
