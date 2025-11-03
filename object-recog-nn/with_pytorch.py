"""Contains the PyTorch implementation of the same L-layer Neural Network."""

import torch.nn as nn


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
