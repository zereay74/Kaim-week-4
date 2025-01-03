# Customer Behavior Analysis

## Project Overview

This project analyzes customer behavior across retail stores using historical sales data. The analysis focuses on how factors like promotions, holidays, and competitor dynamics influence sales performance. Insights are provided to optimize store strategies.

## Features

- **Data Preparation**: A script to load, clean, and transform raw data.
- **Automated Analysis**: A Python class to analyze various aspects of customer behavior and sales trends.
- **Visualization**: Generates plots to present key findings.

## Scripts

- `data_load_clean_transform.py`: Handles data loading, cleaning, and transformation.
- `analysis.py`: Contains the `CustomerBehaviorAnalysis` class for performing analyses and generating plots.

### CustomerBehaviorAnalysis Class
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

## Key Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

For the full list of dependencies, see `requirements.txt`.

## Results

The analysis provides insights into:

- Promotion effectiveness.
- Holiday sales trends.
- Seasonal customer behavior.
- Competitor influence on sales.

