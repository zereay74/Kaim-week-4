# Customer Behavior Analysis

## Project Overview

This project analyzes customer behavior across retail stores using historical sales data. The analysis focuses on how factors like promotions, holidays, and competitor dynamics influence sales performance. Insights are provided to optimize store strategies.

## Features

### Task 1: Exploration

- **Data Preparation**: A script to load, clean, and transform raw data.
- **Automated Analysis**: A Python class to analyze various aspects of customer behavior and sales trends.
- **Visualization**: Generates plots to present key findings.

#### Scripts

- `data_load_clean_transform.py`: Handles data loading, cleaning, and transformation.
- `analysis.py`: Contains the `CustomerBehaviorAnalysis` class for performing analyses and generating plots.

#### CustomerBehaviorAnalysis Class
The `CustomerBehaviorAnalysis` class in `analysis.py` is the core analytical tool for this project. It provides:

- **Functionality:**
  - Analyze sales trends during holidays and promotions.
  - Explore correlations between sales, customer counts, and other factors.
  - Examine seasonal and competitor-related influences on sales.

- **Methods:**
  - `promo_distribution_analysis`: Compares promotional distributions between datasets.
  - `holiday_sales_analysis`: Investigates sales trends before, during, and after holidays.
  - `seasonal_behavior_analysis`: Identifies seasonal patterns in sales.
  - `sales_customer_correlation`: Evaluates the relationship between sales and customer numbers.
  - `promo_effectiveness_analysis`: Measures the impact of promotions on sales and customer engagement.
  - Additional methods for competitor and assortment analysis.

- **Outputs:**
  - Detailed insights into customer and sales behaviors.
  - Clear, customizable plots for each analytical step.

### Task 2: Model Prediction

Building on the exploratory analysis, this task focuses on sales prediction using various regression and machine learning models. Historical sales data is used to predict future trends and assess model performance.

#### Models Implemented

1. **Linear Regression**:
   - Simple and interpretable baseline model.
   - Evaluates relationships between features and sales.

2. **Random Forest**:
   - Handles nonlinear relationships and feature importance.
   - Provides robust predictions by aggregating multiple decision trees.

3. **XGBoost**:
   - Gradient boosting model optimized for high performance.
   - Tackles overfitting and improves accuracy with fine-tuned hyperparameters.

4. **LSTM (Long Short-Term Memory)**:
   - Deep learning model designed for time-series data.
   - Utilizes past sales trends to make sequential predictions.
   - Trained on Kaggle GPU for faster computation.

### Task 3: Modularization for model deployment .... ongoing

#### Workflow

1. **Data Preprocessing**:
   - Split datasets into training, validation, and testing sets.
   - Normalize numerical features and one-hot encode categorical features.
   - Scale target sales for improved model convergence.

2. **Model Training**:
   - Implemented and trained each model using processed data.
   - Fine-tuned hyperparameters for optimal performance.

3. **Evaluation Metrics**:
   - Models were assessed using:
     - Mean Absolute Error (MAE)
     - Root Mean Squared Error (RMSE)
     - R-squared (R²)

4. **Sales Prediction**:
   - Predicted actual sales vs. predicted sales.
   - Generated visualizations for the last 60 days of predictions.

5. **Future Sales Prediction**:
   - Forecasted sales for the next 30 days using LSTM.

#### Outputs

- Comparative performance metrics for all models.
- Plots showing actual vs. predicted sales.
- Future sales forecast visualization.

## Key Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- tensorflow
- keras

For the full list of dependencies, see `requirements.txt`.

## Results

The modeling task provided:

- A comparison of linear regression, random forest, XGBoost, and LSTM for sales prediction.
- Insight into the suitability of each model for time-series forecasting.
- Visualizations to assess model performance and future trends.

By combining insights from Task 1 and Task 2, this project delivers actionable insights for retail sales optimization.

