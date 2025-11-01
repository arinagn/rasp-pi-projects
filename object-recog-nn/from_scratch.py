"""Helper functions for building an L-layer Neural Network from scratch with numpy."""

import numpy as np

np.random.seed(42)


def _sigmoid(Z):
    """Computes the sigmoid activation function.

    Args:
        Z: Input numpy array.

    Returns:
        The sigmoid of Z, where each element is 1 / (1 + e^(-z)).
    """
    return 1 / (1 + np.exp(-Z))


def _relu(Z):
    """Computes the ReLU (Rectified Linear Unit) activation function.

    Args:
        Z: Input numpy array.

    Returns:
        The ReLU of Z, where negative values are replaced with 0.
    """
    return np.maximum(0, Z)


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
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):  # noqa: E741
        parameters["W" + str(l)] = (
            np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        )
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


def linear_forward(A, W, b):
    """Implements the linear step of a layer's forward propagation.

    For a layer l, and all m training examples (vectorised), performs:
    Z^[l] = W^[l] A^[l-1] + b^[l]

    Args:
        A: Activations from previous layer (or input data, i.e. A^[0] = X)
           Dimensions are (# units in previous layer, # training examples)
        W: Weights matrix - a numpy array of a layer l's weights
           Dimensions are (# units in current layer, # units in previous layer)
        b: Bias vector - a numpy array of a layer l's biases
           Dimensions are (# units in the current layer, 1). If all m training
           examples are being used, Python performs broadcasting to compute
           element-wise sum between the bias and each of m training examples.

    Returns:
        Z: The input of the activation function, also called pre-activation parameter
        cache: A python tuple containing "A", "W" and "b";
               stored for computing the backward pass efficiently
    """
    Z = np.dot(W, A) + b
    cache = (A, W, b)

    return Z, cache


def linear_and_activation_forward(A_prev, W, b, activation):
    """Implements forward propagation for layer l.

    Combines the linear step perfomed in a unit with the non-linear activation.

    Args:
        A_prev: Activations from previous layer (or input data, i.e. A^[0] = X)
           Dimensions are (# units in previous layer, # training examples)
        W: Weights matrix - a numpy array of a layer l's weights
           Dimensions are (# units in current layer, # units in previous layer)
        b: Bias vector - a numpy array of a layer l's biases
           Dimensions are (# units in the current layer, 1). If all m training
           examples are being used, Python performs broadcasting to compute
           element-wise sum between the bias and each of m training examples.
        activation: A string indicating the type of activation to be used in this
                    layer, e.g. "sigmoid" or "relu"

    Returns:
    A: A numpy array containing the output of the activation function,
       also called the post-activation value.
    cache: A tuple containing "linear_cache" and "activation_cache";
           stored for computing the backward pass efficiently.
    """
    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == "sigmoid":
        A, activation_cache = _sigmoid(Z)

    elif activation == "relu":
        A, activation_cache = _relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache
