
"""
Neural Networks for Handwritten Digit Recognition (Binary)

Course: Machine Learning Specialization
Frameworks: TensorFlow, Keras, NumPy

Exercises:
1. TensorFlow Neural Network (UNQ_C1)
2. NumPy Dense Layer (UNQ_C2)
3. Vectorized NumPy Dense Layer (UNQ_C3 - Optional)
"""

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# ==========================================
# Activation Function
# ==========================================

def sigmoid(z):
    """Computes the sigmoid activation function."""
    return 1 / (1 + np.exp(-z))


# ==========================================
# Exercise 1: TensorFlow Model (UNQ_C1)
# ==========================================

def create_model():
    """
    Creates a three-layer neural network using TensorFlow/Keras.

    Architecture:
        Input layer: 400 features
        Hidden layer 1: 25 units, sigmoid
        Hidden layer 2: 15 units, sigmoid
        Output layer: 1 unit, sigmoid

    Returns:
        model: Compiled-ready Keras Sequential model.
    """

    model = Sequential(
        [
            tf.keras.Input(shape=(400,)),
            Dense(25, activation='sigmoid'),
            Dense(15, activation='sigmoid'),
            Dense(1, activation='sigmoid')
        ],
        name="my_model"
    )

    return model


# ==========================================
# Exercise 2: NumPy Dense Layer (UNQ_C2)
# ==========================================

def my_dense(a_in, W, b, g):
    """
    Computes a dense layer for one example.

    Args:
        a_in: Input vector (n_features,)
        W: Weight matrix (n_features, n_units)
        b: Bias vector (n_units,)
        g: Activation function

    Returns:
        a_out: Output vector (n_units,)
    """

    units = W.shape[1]
    a_out = np.zeros(units)

    for j in range(units):
        z = np.dot(W[:, j], a_in) + b[j]
        a_out[j] = g(z)

    return a_out


# ==========================================
# Three-Layer NumPy Neural Network
# ==========================================

def my_sequential(x, W1, b1, W2, b2, W3, b3):
    """
    Implements a three-layer neural network using NumPy.
    """

    a1 = my_dense(x, W1, b1, sigmoid)
    a2 = my_dense(a1, W2, b2, sigmoid)
    a3 = my_dense(a2, W3, b3, sigmoid)

    return a3


# ==========================================
# Exercise 3: Vectorized Dense Layer (UNQ_C3)
# Optional Exercise
# ==========================================

def my_dense_v(A_in, W, b, g):
    """
    Computes a vectorized dense layer for multiple examples.

    Args:
        A_in: Input matrix (m, n_features)
        W: Weight matrix (n_features, n_units)
        b: Bias vector (n_units,)
        g: Activation function

    Returns:
        A_out: Output matrix (m, n_units)
    """

    A_out = g(np.matmul(A_in, W) + b)

    return A_out


# ==========================================
# Vectorized Three-Layer Neural Network
# ==========================================

def my_sequential_v(X, W1, b1, W2, b2, W3, b3):
    """
    Implements a vectorized three-layer neural network.
    Processes multiple examples simultaneously.
    """

    A1 = my_dense_v(X, W1, b1, sigmoid)
    A2 = my_dense_v(A1, W2, b2, sigmoid)
    A3 = my_dense_v(A2, W3, b3, sigmoid)

    return A3


# ==========================================
# Prediction Helper
# ==========================================

def predict_digit(probability):
    """
    Converts predicted probability into a binary class.

    0 = Handwritten digit zero
    1 = Handwritten digit one
    """

    return (probability >= 0.5).astype(int)
