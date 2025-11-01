"""Functionality for image processing, built on top of opencv-python."""

import numpy as np
import cv2
import os


IMG_SIZE = 64

def load_images(folder, label):
    """Loads images from a given folder, resizes, flattens."""
    X, Y = [], []
    for file in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, file))
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X.append(img.flatten())
        Y.append(label)
    return np.array(X), np.array(Y)


def create_data(folder_pos, folder_neg):
    """Create training data from loaded images."""
    X_pos, Y_pos = load_images(folder_pos, 1)
    X_neg, Y_neg = load_images(folder_neg, 0)
    X = np.concatenate((X_pos, X_neg), axis=0).T
    X = X / 255.0
    Y = np.concatenate((Y_pos, Y_neg), axis=0).reshape[1, -1]

    return X, Y