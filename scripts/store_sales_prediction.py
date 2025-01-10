import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.utils import resample
import shap
import joblib
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class StoreSalesPredictor:
    def __init__(self, store_data=None, train_data=None, test_data=None):
        logging.basicConfig(level=logging.INFO)
        self.store_data = store_data
        self.train_data = train_data
        self.test_data = test_data
        self.pipeline = None
        self.scaler = StandardScaler()

    def preprocess_data(self):
        logging.info("Preprocessing train and test data.")
        # Convert dates
        self.train_data['Date'] = pd.to_datetime(self.train_data['Date'])
        self.test_data['Date'] = pd.to_datetime(self.test_data['Date'])

        # Feature extraction
        for df in [self.train_data, self.test_data]:
            df['Weekday'] = df['Date'].dt.weekday
            df['Weekend'] = df['Weekday'] >= 5
            df['Month'] = df['Date'].dt.month
            df['Day'] = df['Date'].dt.day
            df['StartOfMonth'] = df['Day'] <= 10
            df['MidMonth'] = (df['Day'] > 10) & (df['Day'] <= 20)
            df['EndOfMonth'] = df['Day'] > 20

        # Merge store_data with train and test data
        self.train_data = self.train_data.merge(self.store_data, on='Store', how='left')
        self.test_data = self.test_data.merge(self.store_data, on='Store', how='left')

        # Scale numeric features
        numeric_features = ['CompetitionDistance']
        self.train_data[numeric_features] = self.scaler.fit_transform(self.train_data[numeric_features])
        if numeric_features[0] in self.test_data.columns:
            self.test_data[numeric_features] = self.scaler.transform(self.test_data[numeric_features])

    def train_model(self):
        logging.info("Training Random Forest model.")

        # Prepare data
        X = self.train_data.drop(columns=['Sales', 'Date', 'StateHoliday'])
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
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])

        # Train the pipeline
        self.pipeline.fit(X_train, y_train)

        # Validate the model
        y_pred = self.pipeline.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        logging.info(f"Validation MAE: {mae}")
        logging.info(f"Validation MSE: {mse}")
    def plot_shap_importance(self):
        logging.info("Calculating SHAP feature importance.")
        
        # Prepare the data
        X = self.train_data.drop(columns=['Sales', 'Date', 'StateHoliday'])
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        numeric_features = X.select_dtypes(include=['number']).columns
        preprocessor = self.pipeline.named_steps['preprocessor']
        
        # Preprocess the data
        X_preprocessed = preprocessor.transform(X)
        
        # Get feature names
        feature_names = (
            preprocessor.named_transformers_['cat']
            .get_feature_names_out(categorical_features)
            .tolist()
            + numeric_features.tolist()
        )
        
        # SHAP explainer
        explainer = shap.Explainer(self.pipeline.named_steps['regressor'], X_preprocessed)
        shap_values = explainer(X_preprocessed)
        
        # Plot SHAP summary
        shap.summary_plot(shap_values, feature_names=feature_names, show=False)
        plt.title("SHAP Feature Importance")
        plt.show()


    def estimate_confidence_intervals(self, X_val, num_samples=100):
        logging.info("Estimating confidence intervals using bootstrapping.")
        predictions = []
        for _ in range(num_samples):
            X_resampled, y_resampled = resample(self.train_data.drop(columns=['Sales']), self.train_data['Sales'])
            self.pipeline.fit(X_resampled, y_resampled)
            predictions.append(self.pipeline.predict(X_val))

        predictions = np.array(predictions)
        lower_bound = np.percentile(predictions, 2.5, axis=0)
        upper_bound = np.percentile(predictions, 97.5, axis=0)
        return lower_bound, upper_bound

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
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(time_steps, 1)),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=10, batch_size=32)
        return model

    def plot_predictions(self, model_type='random_forest', X_val=None, y_val=None, lstm_model=None, time_steps=10):
        logging.info("Plotting predictions.")
        if model_type == 'random_forest':
            y_pred = self.pipeline.predict(X_val)
            plt.figure(figsize=(10, 6))
            plt.plot(y_val[:100], label="True Sales")
            plt.plot(y_pred[:100], label="Predicted Sales")
            plt.title("Random Forest Predictions")
            plt.legend()
            plt.show()
        elif model_type == 'lstm':
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

            y_pred = lstm_model.predict(X)
            plt.figure(figsize=(10, 6))
            plt.plot(y[:100], label="True Sales")
            plt.plot(y_pred[:100], label="Predicted Sales")
            plt.title("LSTM Predictions")
            plt.legend()
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
