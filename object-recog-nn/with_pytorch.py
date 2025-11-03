"""Contains the PyTorch implementation of the same L-layer Neural Network."""

import torch.nn as nn
import torch.optim as optim
from dnn_utils import LAYER_DIMS


class LLayerNN(nn.Module):
    """Defines a PyTorch based L-layer Neural Network."""

    def __init__(self, layer_dims):
        """layer_dims: A tuple specifying # of units in each layer."""
        super(LLayerNN, self).__init__()

        layers = []
        L = len(layer_dims)

        for l in range(1, L):  # noqa: E741
            in_dim = layer_dims[l - 1]
            out_dim = layer_dims[l]

            layers.append(nn.Linear(in_features=in_dim, out_features=out_dim))

            if l < L - 1:
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Sigmoid())

        # Define Sequential model where all layers stacked
        self.network = nn.Sequential(*layers)

    def forward(self, X):
        """Performs forward prop, linear -> activation, across L layers.

        Args:
            X: Input feature matrix of shape (m, n_x). If only one training example
               is used (m=1, non-vectorised), this is an input feature vector.
               Note that PyTorch prefers one row per training example instead of
               the notation used in ML mathematics  where rows represent a feature
               vector for all m training examples.
        """
        return self.network(X)


def L_layer_nn_model_torch(
    X,
    Y,
    layer_dims=LAYER_DIMS,
    learning_rate=0.01,
    num_iterations=10000,
    print_cost=False,
):
    """Implements a L-layer neural network with PyTorch.

    Follows the same architecture as the NN built from scratch;
    [LINEAR->RELU]*[L-1] -> [LINEAR->SIGMOID].

    The L-layer NN implemented from scratch used numpy arrays for
    efficient matrix / vector computations. PyTorch uses tensors for
    storing data, from input matrices, to weights and biases.
    Computations with PyTorch tensors can be run significantly faster
    on CUDA-compatible GPUs.

    Args:
        X: Input feature tensor of shape (# training examples, n_x)
        Y: True label tensor (0 / 1) of shape (# training examples, 1)
        layer_dims: List / tuple containing the number of units in each
                     layer, e.g. (n_x, n_h1, n_h2, n_y)
        learning_rate: A scalar used in the gradient descent update rule
        num_iterations: Number of iterations of the optimization loop
        print_cost: If True cost printed every 100 steps
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = LLayerNN(layer_dims).to(device)

    X = X.to(device)
    Y = Y.to(device)

    criterion = nn.BCELoss()
    optimiser = optim.SGD(model.parameters(), lr=learning_rate)

    for i in range(num_iterations):
        AL = model(X)  # equivalent to l_layer_model_forward()
        # Note the use of model(X) instead of calling the forward method
        # directly as model.forward(X)
        # PyTorch's nn.module overloads the __call__ method that is run
        # when executing model(X) => self.forward() called inside __call__

        loss = criterion(AL, Y)  # compute_cost()
        optimiser.zero_grad()  # clear prev gradients
        loss.backward()  # l_layer_model_backward()
        optimiser.step()  # update_parameters()

        if print_cost and (i % 100 == 0 or i == num_iterations - 1):
            print(f"Cost after iteration {i}: {loss.item():.6f}")


if __name__ == "__main__":
    import torch

    X = torch.randn(100, 12288)
    Y = torch.randint(0, 2, (100, 1)).float()

    L_layer_nn_model_torch(X, Y, print_cost=True)
