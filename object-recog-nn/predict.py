"""Predicts the class of a new input image using the trained NN."""

import cv2
import numpy as np
from dnn_utils import linear_and_activation_forward
from image_preprocess import IMG_SIZE


def predict(image, parameters):
    """Computes predictions by running forward prop using updated parameters.

    Args:
        image: A .jpg image you want to classify. This will be resized, flattened,
               and normalised using openCV before a prediction is made.
        parameters: A dictionary of parameters optimised by the model via
                    gradient descent in the back propagation step.
    """
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE)).flatten().reshape(-1, 1) / 255.0
    A2, _ = linear_and_activation_forward(image, parameters)
    return 1 if A2 > 0.5 else 0


if __name__ == "__main__":
    model = np.load("nn_params.npz")
    params = {key: model[key] for key in model}

    cam = cv2.VideoCapture(0)
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        pred = predict(frame, params)
        label = "POM" if pred == 1 else "NO POM"
        cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Pi Camera Classifier", frame)
        if cv2.waitKey(1) == ord("q"):
            break
    cam.release()
    cv2.destroyAllWindows()
