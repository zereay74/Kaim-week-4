# Data Manipulation Libraries
import pandas as pd
import numpy as np
# Plotting Libraries
from matplotlib import pyplot as plt
import seaborn as sns
# Machine Learning Libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
# Deep Learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.models import load_model

import os
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.utils import register_keras_serializable

def load_model_with_custom_metric(model_path):
    """
    Loads a Keras model with a custom Mean Squared Error (MSE) metric.

    Args:
        model_path (str): Path to the .h5 model file.

    Returns:
        model: Loaded Keras model with custom MSE metric.
    """
    # Define custom MSE metric
    @register_keras_serializable()
    def mse(y_true, y_pred):
        return MeanSquaredError()(y_true, y_pred)

    # Check if the model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model file at {model_path} does not exist.")

    # Load the model with the custom MSE metric
    model = load_model(model_path, custom_objects={'mse': mse})
    print(f"Model loaded successfully from {model_path}")
    return model
