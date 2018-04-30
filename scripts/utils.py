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

def plot_layer_activations(image, model):
    #     #Create new sequential model, same as before but just keep the convolutional layer.
    # model_new = Sequential()
    # model_new.add(Conv2D(nb_filters, kernel_size=(3, 3),
    #                  activation='relu',
    #                  input_shape=input_shape))
    # #set weights for new model from weights trained on MNIST.
    # for i in range(1):
    #     model_new.layers[i].set_weights(model.layers[i].get_weights())
    # #pick a random digit and "predict" on this digit (output will be first layer of CNN)
    # i = np.random.randint(0,len(x_test))
    # digit = x_test[i].reshape(1,28,28,1)
    # pred = model_new.predict(digit)
    # #check shape of prediction
    # print pred.shape
    # (1, 26, 26, 32)
    # #For all the filters, plot the output of the input
    # plt.figure(figsize=(18,18))
    # filts = pred[0]
    # for i in range(nb_filters):
    #     filter_digit = filts[:,:,i]
    #     plt.subplot(6,6,i+1)
    #     plt.imshow(filter_digit,cmap='gray'); plt.axis('off');

    pass

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
