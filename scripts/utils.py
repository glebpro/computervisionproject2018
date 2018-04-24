#
#   Some general utilities to use throughout project.
#
#   @author Gleb Promokhov
#

import pathlib
import os

from matplotlib import pyplot as plt
import numpy as np
import cv2


def plot_saliency(image, model):
    # model.layers[layer_idx].activation = activations.linear
    # model = utils.apply_modifications(model)

    x_test = cv2.imread(image)
    x_test = cv2.resize(x_test, (150, 150))

    grads = []
    titles = []
    for layer_idx in list(range(len(model.layers))):
        titles.append(model.layers[layer_idx].name)
        grads.append(visualize_saliency(model, layer_idx, filter_indices=1, seed_input=x_test))

    show_images(grads, titles, 4)

def show_images(images, titles = None, cols = 1):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        plt.imshow(image)
        plt.axis('off')
        a.set_title(title)

    # fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.tight_layout()
    plt.show()

__all__ = ['show_images', 'plot_saliency']
