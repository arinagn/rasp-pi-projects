"""MLflow experiment logging parameters, metrics, and models."""

import json
import mlflow
import os
import numpy as np
import time
from from_scratch import L_layer_nn_model


def run_one_experiment(X, Y, layer_dims, lr, num_iterations, experiment="numpy-nn"):
    """MLflow experiment for logging hyper-params, metrics, and model for inference."""
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=f"hd={layer_dims}_lr={lr}"):
        # Log hyperparameters for this run
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("num_iterations", num_iterations)
        mlflow.log_param("layer_dims", json.dumps(layer_dims))

        params, costs = L_layer_nn_model(
            X=X,
            Y=Y,
            layers_dims=layer_dims,
            learning_rate=lr,
            num_iterations=num_iterations,
        )

        # Log metrics computed at each step
        for step, loss in enumerate(costs):
            mlflow.log_metric("loss", float(loss), step=step)

        # Store learnt parameters in artifacts/
        os.makedirs("artifacts", exist_ok=True)
        np.savez("artifacts/params.npz", **params)

        # Plot loss curve over iterations
        try:
            import matplotlib.pyplot as plt

            plt.figure()
            plt.plot(costs)
            plt.xlabel("iteration")
            plt.ylabel("loss")
            plt.title("Training loss")
            plot_path = "artifacts/loss.png"
            plt.savefig(plot_path, bbox_inches="tight")
            plt.close()
            mlflow.log_artifact(plot_path)
        except Exception:
            pass

        mlflow.log_artifact("artifacts/params.npz")

        # Log final loss for easy sorting
        mlflow.log_metric("final_loss", float(costs[-1]))

        # Optional tags
        mlflow.set_tag("framework", "numpy")
        mlflow.set_tag("run_ts", time.strftime("%Y-%m-%d %H:%M:%S"))

        return params, costs


def mlflow_sweep(X, Y):
    """Defines grid-search over learning rate and NN architecture.

    The output is logged to MLflow Tracking Server on localhost.
    """
    lr_options = [0.001, 0.01, 0.05]
    layer_dims_options = [
        [12288, 1000, 1],
        [12288, 50, 2, 1],
        [12288, 10, 6, 4, 1],
    ]
    iters = 10

    for layer_dims_option in layer_dims_options:
        for lr in lr_options:
            run_one_experiment(X, Y, layer_dims_option, lr, iters, experiment="numpy-nn")
