"""Predicts the class of a new input image using the trained NN."""

import cv2
import numpy as np
from picamera2 import Picamera2
from dnn_utils import l_layer_model_forward
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
    A2, _ = l_layer_model_forward(image, parameters)
    return 1 if A2 > 0.5 else 0


if __name__ == "__main__":

    # Load learnt model parameters
    model = np.load("nn_params.npz")
    params = {key: model[key] for key in model}

    # Acquire camera device and initialise libcamera components
    picam2 = Picamera2()

    # Ensure RGB format so that 3 channels => (64, 64, 3)
    camera_config = picam2.create_preview_configuration(main={"format": "RGB888"})
    picam2.configure(camera_config)
    picam2.start()

    try:
        while True:
            frame = picam2.capture_array()
            print(frame.shape)
            pred = predict(frame, params)
            label = "POM" if pred==1 else "NO POM"

            cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Classifier", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(1) == ord("q"):
                break
    finally:
        picam2.stop()
        picam2.close()
        cv2.destroyAllWindows()
