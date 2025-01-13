import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
import shap
import joblib
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class StoreSalesPredictor:
    def __init__(self, train_data=None, test_data=None):
        logging.basicConfig(level=logging.INFO)
        self.train_data = train_data
        self.test_data = test_data
        self.pipeline = None
        self.lstm_model = None
        self.scaler = StandardScaler()

    def preprocess_data(self):
        logging.info("Preprocessing data.")
        # Extract date features
        for df in [self.train_data, self.test_data]:
            df['Date'] = pd.to_datetime(df['Date'])
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['Week'] = df['Date'].dt.isocalendar().week
            df['Weekday'] = df['Date'].dt.weekday
            df['Weekend'] = (df['Weekday'] >= 5).astype(int)
            df['Day'] = df['Date'].dt.day

        # Fill missing values
        self.train_data.fillna(self.train_data.median(), inplace=True)
        self.test_data.fillna(self.test_data.median(), inplace=True)

    def train_model(self):
        logging.info("Training XGBoost model.")

        # Prepare data
        X = self.train_data.drop(columns=['Sales', 'Date'])
        y = self.train_data['Sales']
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Identify categorical and numeric features
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        numeric_features = X.select_dtypes(include=['number']).columns

        # Create a preprocessor
        preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

        # Create pipeline
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', XGBRegressor(objective='reg:squarederror', random_state=42))
        ])

        # Hyperparameter tuning
        param_grid = {
            'regressor__n_estimators': [100, 200],
            'regressor__max_depth': [3, 5],
            'regressor__learning_rate': [0.01, 0.1]
        }
        grid_search = GridSearchCV(self.pipeline, param_grid, cv=3, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)

        # Set the best model
        self.pipeline = grid_search.best_estimator_

        # Validate the model
        y_pred = self.pipeline.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        logging.info(f"Validation MAE: {mae}")
        logging.info(f"Validation MSE: {mse}")

    def plot_shap_importance(self):
        logging.info("Calculating SHAP feature importance.")
        X = self.train_data.drop(columns=['Sales', 'Date'])

        explainer = shap.Explainer(self.pipeline.named_steps['regressor'], self.pipeline.named_steps['preprocessor'].transform(X))
        shap_values = explainer.shap_values(self.pipeline.named_steps['preprocessor'].transform(X))

        # Plot SHAP summary
        shap.summary_plot(shap_values, self.pipeline.named_steps['preprocessor'].transform(X))
        plt.title("SHAP Feature Importance")
        plt.show()

    def save_model(self):
        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        model_filename = f"model_{timestamp}.pkl"
        joblib.dump(self.pipeline, model_filename)
        logging.info(f"Model saved as {model_filename}")

    def load_model(self, model_filename):
        logging.info(f"Loading model from {model_filename}.")
        self.pipeline = joblib.load(model_filename)
        logging.info("Model loaded successfully.")

    def build_lstm_model(self, time_steps=10):
        logging.info("Building LSTM model.")
        data = self.train_data[['Date', 'Sales']].copy()
        data.sort_values(by='Date', inplace=True)
        sales_series = data['Sales'].values

        # Prepare data for LSTM
        X, y = [], []
        for i in range(len(sales_series) - time_steps):
            X.append(sales_series[i:i + time_steps])
            y.append(sales_series[i + time_steps])
        X, y = np.array(X), np.array(y)

        # Scale data
        X = self.scaler.fit_transform(X)
        y = self.scaler.transform(y.reshape(-1, 1)).flatten()

        # Reshape for LSTM
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Build LSTM model
        self.lstm_model = Sequential([
            LSTM(50, activation='relu', input_shape=(time_steps, 1)),
            Dense(1)
        ])

        self.lstm_model.compile(optimizer='adam', loss='mse')
        self.lstm_model.fit(X, y, epochs=10, batch_size=32)

    def plot_predictions(self, X_val=None, y_val=None):
        logging.info("Plotting predictions.")
        y_pred = self.pipeline.predict(X_val)
        plt.figure(figsize=(10, 6))
        plt.plot(y_val[:100], label="True Sales")
        plt.plot(y_pred[:100], label="Predicted Sales")
        plt.title("XGBoost Predictions")
        plt.legend()
        plt.show()

    def plot_lstm_predictions(self, time_steps=10):
        logging.info("Plotting LSTM predictions.")
        data = self.train_data[['Date', 'Sales']].copy()
        data.sort_values(by='Date', inplace=True)
        sales_series = data['Sales'].values

        # Prepare data for LSTM
        X, y = [], []
        for i in range(len(sales_series) - time_steps):
            X.append(sales_series[i:i + time_steps])
            y.append(sales_series[i + time_steps])
        X, y = np.array(X), np.array(y)

        # Scale data
        X = self.scaler.transform(X)
        y = self.scaler.transform(y.reshape(-1, 1)).flatten()

        # Reshape for LSTM
        X = X.reshape((X.shape[0], X.shape[1], 1))

        y_pred = self.lstm_model.predict(X)
        plt.figure(figsize=(10, 6))
        plt.plot(y[:100], label="True Sales")
        plt.plot(y_pred[:100], label="Predicted Sales")
        plt.title("LSTM Predictions")
        plt.legend()
        plt.show()
''' 
# Usage example
if __name__ == "__main__":
    # Assume train_data and test_data are preloaded Pandas DataFrames
    predictor = StoreSalesPredictor(train_data=train_data, test_data=test_data)
    predictor.preprocess_data()
    predictor.train_model()
    predictor.plot_shap_importance()
    predictor.save_model()

    # For LSTM
    predictor.build_lstm_model()
    predictor.plot_lstm_predictions()
'''