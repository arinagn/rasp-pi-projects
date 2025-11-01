"""Helper functions for building an L-layer Neural Network from scratch with numpy."""

import numpy as np

np.random.seed(42)


def initialize_parameters_l_layers(layer_dims):
    """Initialises weight and bias parameters for each layer l in the network.

    Args:
        layer_dims: An array containing the dimensions of each layer in our network.
                    e.g. (n_x, n_h, n_y) where n_x denotes the number of units in the 
                    input layer (l=0), n_h is the number of units in the hidden layer 
                    (l=1) and n_y is the number of units in the output layer of this 
                    2 layer neural network.
    
    Returns:
        parameters: A dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl - weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl - bias vector of shape (layer_dims[l], 1)
    """
    parameters = {}
    L = len(layer_dims) # number of layers in the network

    for l in range(1, L): # noqa: E741
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

