import numpy as np
import warnings

from utiles import *

warnings.simplefilter('ignore')
import matplotlib.pyplot as plt
import subprocess
import tensorflow as tf


# tf.nn.avg_pool2d
# tf.nn.avg_pool3d

def local_response_norm(input, size, alpha=1e-4, beta=0.75, k=1):
    r"""Applies local response normalization over an input signal composed of
    several input planes, where channels occupy the second dimension.
    Applies normalization across channels.
    See :class:`~torch.nn.LocalResponseNorm` for details.
    >>> dataset = get_quick_dataset(inRGB=False)
    >>> dataset = [i.transpose((2, 0, 1)) for i in dataset]
    >>> res = local_response_norm(dataset)
    >>> a = plt.imshow(res[3, :, :, :], cmap="gray"); a = plt.axis('off'); a = plt.colorbar()
    >>> plt.show()
    """
    dim = input.dim()
    if dim < 3:
        raise ValueError('Expected 3D or higher dimensionality \
                         input (got {} dimensions)'.format(dim))
    div = (input * input).unsqueeze(1)
    if dim == 3:
        paddings = [[size // 2, (size - 1) // 2], [0, 0]]
        div = tf.pad(div, paddings)
        div = tf.nn.avg_pool2d(div, (size, 1), stride=1).squeeze(1)
    else:
        sizes = input.size()
        div = div.view(sizes[0], 1, sizes[1], sizes[2], -1)
        paddings = [[size // 2, (size - 1) // 2], [0, 0], [0, 0]]
        div = tf.pad(div, paddings)
        div = tf.nn.avg_pool3d(div, (size, 1, 1), stride=1).squeeze(1)
        div = div.view(sizes)
    div = (div * alpha + k) ** beta
    return input / div


# images_list = subprocess.getoutput('ls images/').split('\n')
# images = [plt.imread('images/' + i) for i in images_list]
#
# images = [i.transpose((2, 0, 1)) for i in images]

# images_tensor = tf.Tensor([i for i in images])

from PIL import Image as pim

# test = local_response_norm(torch.Tensor([images_tensor[2].numpy()]),3)

# test.max(),test.min()
