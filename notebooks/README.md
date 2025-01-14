
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

  ### Task 3: Modularization for model deployment
  