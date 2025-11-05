"""Logs and registers a numpy-based Neural Network to MLflow Model Registry."""

import cv2
import numpy as np
import mlflow
import mlflow.pyfunc
from dnn_utils import l_layer_model_forward
from image_preprocess import IMG_SIZE


class ScratchNeuralNet(mlflow.pyfunc.PythonModel):
    """Wrapping the logic of the numpy model with a custom mlflow.pyfunc.PythonModel.

    Since the NN built from scratch was developed using numpy, there is no
    dedicated MLflow flavour (e.g. mlflow.sklearn, or mlflow.pytorch) to handle
    the model logging process automatically via mlflow.<flavour>.log_model().

    The pyfunc wrapper allows MLflow to serialise the model and its dependencies,
    and the log it to MLflow Model Registry.

    The reason you cannot just use a mlflow.log_model() similar to mlflow.log_param()
    or mlflow.log_artifact() we saw in MLflow's Tracking Server is becuase of the
    difference in tracking artifacts and logging a deployable model. When you run
    mlflow.log_artifact("params.npz"), the MLflow simply takes the npz file and
    uploads it to the MLflow backend. MLflow does not know how to turn the params.npz
    file into a model that can accept input and return a prediction.

    The mlflow.pyfunc.PythonModel wrapper tells MLflow how to do these 2 things:
    1. How to load and use the saved parameters via the load_context() method.
    2. How to make a prediction via the predict() method. This is the inference
    function that will be called every time a prediction is requested. In this
    case it will be the L-layer forward propogration logic.
    """

    def load_context(self, context):
        """Tells MLflow how to load the param file and use to initalise model."""
        params = np.load(context.artifacts["params_path"])
        self.params = {key: params[key] for key in params}

    def predict(self, context, model_input):
        """The inference function to be called when a pred is requested."""
        image = (
            cv2.resize(model_input, (IMG_SIZE, IMG_SIZE)).flatten().reshape(-1, 1)
            / 255.0
        )
        A2, _ = l_layer_model_forward(image, self.params)
        return 1 if A2 > 0.5 else 0


if __name__ == "__main__":
    mlflow.set_experiment("trial_model_registry")
    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            name="scratch-neural-net",
            python_model=ScratchNeuralNet(),
            artifacts={"params_path": "object-recog-nn/nn_params.npz"},
            registered_model_name="scratch-neural-net",
        )
