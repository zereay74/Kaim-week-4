import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CustomerBehaviorAnalysis:
    def __init__(self, store_data, train_data, test_data):
        self.store_data = store_data
        self.train_data = train_data
        self.test_data = test_data
        logging.info("CustomerBehaviorAnalysis instance created.")

    def plot_promo_distribution(self):
        """Check for promo distribution in train and test data."""
        logging.info("Executing plot_promo_distribution.")
        train_promo = self.train_data['Promo'].value_counts(normalize=True)
        test_promo = self.test_data['Promo'].value_counts(normalize=True)
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        sns.barplot(x=train_promo.index, y=train_promo.values, ax=ax[0])
        sns.barplot(x=test_promo.index, y=test_promo.values, ax=ax[1])
        ax[0].set_title('Promo Distribution (Train)')
        ax[1].set_title('Promo Distribution (Test)')
        plt.show()

    def sales_during_holidays(self):
        """Analyze sales behavior before, during, and after holidays."""
        logging.info("Executing sales_during_holidays.")
        self.train_data['IsHoliday'] = self.train_data['StateHoliday'] != '0'
        holiday_sales = self.train_data.groupby('IsHoliday')['Sales'].mean()
        
        holiday_sales.plot(kind='bar', title='Sales During Holidays vs Non-Holidays', figsize=(8, 6))
        plt.xlabel('Holiday Status')
        plt.ylabel('Average Sales')
        plt.show()

    def seasonal_purchase_behavior(self):
        """Find seasonal trends such as Christmas and Easter."""
        logging.info("Executing seasonal_purchase_behavior.")
        self.train_data['Month'] = pd.to_datetime(self.train_data['Date']).dt.month
        seasonal_sales = self.train_data.groupby('Month')['Sales'].mean()
        
        seasonal_sales.plot(kind='line', title='Seasonal Sales Trends', figsize=(10, 6))
        plt.xlabel('Month')
        plt.ylabel('Average Sales')
        plt.show()

    def sales_customer_correlation(self):
        """Find correlation between sales and number of customers."""
        logging.info("Executing sales_customer_correlation.")
        correlation = self.train_data[['Sales', 'Customers']].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Between Sales and Customers')
        plt.show()

    def promo_effectiveness(self):
        """How do promotions affect sales and customers?"""
        logging.info("Executing promo_effectiveness.")
        promo_effect = self.train_data.groupby('Promo')[['Sales', 'Customers']].mean()
        promo_effect.plot(kind='bar', figsize=(12, 6), title='Promo Effect on Sales and Customers')
        plt.xlabel('Promo')
        plt.ylabel('Average Values')
        plt.show()

    def store_open_days_analysis(self):
        """Analyze sales for stores open on all weekdays vs weekends."""
        logging.info("Executing store_open_days_analysis.")
        weekday_sales = self.train_data[self.train_data['DayOfWeek'] < 6].groupby('Store')['Sales'].mean()
        weekend_sales = self.train_data[self.train_data['DayOfWeek'] >= 6].groupby('Store')['Sales'].mean()
        
        combined_sales = pd.DataFrame({'WeekdaySales': weekday_sales, 'WeekendSales': weekend_sales})
        combined_sales.plot(kind='scatter', x='WeekdaySales', y='WeekendSales', alpha=0.7, figsize=(10, 6))
        plt.title('Weekday vs Weekend Sales by Store')
        plt.xlabel('Weekday Sales')
        plt.ylabel('Weekend Sales')
        plt.show()

    def assortment_sales_analysis(self):
        """Analyze how assortment types affect sales."""
        logging.info("Executing assortment_sales_analysis.")
        merged_data = self.train_data.merge(self.store_data, on='Store')
        assortment_sales = merged_data.groupby('Assortment')['Sales'].mean()
        
        assortment_sales.plot(kind='bar', title='Sales by Assortment Type', figsize=(8, 6))
        plt.xlabel('Assortment')
        plt.ylabel('Average Sales')
        plt.show()

    def competitor_distance_effect(self):
        """Analyze the effect of competitor distance on sales."""
        logging.info("Executing competitor_distance_effect.")
        merged_data = self.train_data.merge(self.store_data, on='Store')
        sns.scatterplot(data=merged_data, x='CompetitionDistance', y='Sales', alpha=0.5)
        plt.title('Competitor Distance vs Sales')
        plt.xlabel('Competition Distance')
        plt.ylabel('Sales')
        plt.show()

    def competitor_opening_analysis(self):
        """Analyze the impact of new competitors opening near stores."""
        logging.info("Executing competitor_opening_analysis.")
        merged_data = self.train_data.merge(self.store_data, on='Store')
        merged_data['HasCompetitor'] = ~merged_data['CompetitionDistance'].isna()
        competitor_sales = merged_data.groupby('HasCompetitor')['Sales'].mean()
        
        competitor_sales.plot(kind='bar', title='Sales Before and After Competitor Opening', figsize=(8, 6))
        plt.xlabel('Competitor Presence')
        plt.ylabel('Average Sales')
        plt.show()

# Usage Example
# analysis = CustomerBehaviorAnalysis(store_data, train_data, test_data)
# analysis.plot_promo_distribution()
# analysis.sales_during_holidays()
# and so on...
