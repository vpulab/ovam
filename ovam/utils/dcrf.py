import denseCRF
import numpy as np
from PIL import Image


def densecrf(I, P, w1=10.0, alpha=80, beta=13, w2=3.0, gamma=3, it=5.0):
    """
    input parameters:
        I    : a numpy array of shape [H, W, C], where C should be 3.
               type of I should be np.uint8, and the values are in [0, 255]
        P    : a probability map of shape [H, W, L], where L is the number of classes
               type of P should be np.float32
        param: a tuple giving parameters of CRF (w1, alpha, beta, w2, gamma, it), where
                w1    :   weight of bilateral term, e.g. 10.0
                alpha :   spatial distance std, e.g., 80
                beta  :   rgb value std, e.g., 15
                w2    :   weight of spatial term, e.g., 3.0
                gamma :   spatial distance std for spatial term, e.g., 3
                it    :   iteration number, e.g., 5
    output parameters:
        out  : a numpy array of shape [H, W], where pixel values represent class indices.
    """
    I = np.array(I, dtype=np.uint8)
    P = np.array(P, dtype=np.float32)
    if len(P.shape) == 2:
        P = np.stack([1 - P, P], axis=-1)
    param = (w1, alpha, beta, w2, gamma, it)
    out = denseCRF.densecrf(I, P, param)
    return out
