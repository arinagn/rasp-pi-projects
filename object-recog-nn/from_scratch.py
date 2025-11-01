"""Uses helper functions from dnn_utils to train a L-layer Neural Network."""

import numpy as np

from dnn_utils import (
    initialize_parameters_l_layers,
    l_layer_model_forward,
    compute_cost,
    l_layer_model_backward,
    update_parameters,
)

from image_preprocess import create_data
import cv2


LAYER_DIMS = [12288, 200, 1]


def L_layer_nn_model(
    X, Y, layers_dims, learning_rate=0.0075, num_iterations=10000, print_cost=False
):
    """Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Args:
        X: Input feature matrix of shape (n_x, # training examples)
        Y: True label vector (0 / 1) of shape (1, # training examples)
        layers_dims: List containing the input size and each layer size of length
                    (number of layers + 1).
        learning_rate: A scalar used in the gradient descent update rule
        num_iterations: Number of iterations of the optimization loop
        print_cost: If True cost printed every 100 steps

    Returns:
        parameters: Parameters learnt by the model, to be used for prediction.
    """
    np.random.seed(42)
    costs = []

    parameters = initialize_parameters_l_layers(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):
        AL, caches = l_layer_model_forward(X, parameters)

        cost = compute_cost(AL, Y)
        grads = l_layer_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and (i % 100 == 0 or i == num_iterations - 1):
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))

        if i % 100 == 0:
            costs.append(cost)

    return parameters, costs


def predict(image, parameters):
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE)).flatten().reshape(-1, 1) / 255.
    A2, _ = linear_and_activation_forward(image, parameters)
    return 1 if A2 > 0.5 else 0


if __name__=="__main__":
    X, Y = create_data(folder_pos="data/pom", folder_neg="data/nopom")
    print(X.shape)
    print(Y.shape)

    params, costs = L_layer_nn_model(
        X=X,
        Y=Y,
        layers_dims=LAYER_DIMS,
        print_cost=True,
    )

    print(params)

    cam = cv2.VideoCapture(0)
    while True:
        ret, frame = cam.read()
        if not ret: break
        pred = predict(frame, parameters)
        label = "POM" if pred == 1 else "NO POM"
        cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Pi Camera Classifier", frame)
        if cv2.waitKey(1) == ord('q'): break
    cam.release()
    cv2.destroyAllWindows()