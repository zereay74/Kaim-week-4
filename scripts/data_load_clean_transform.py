import os 
import sys
import pandas as pd
import psycopg2 
from dotenv import load_dotenv
from sqlalchemy import create_engine
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    A class to handle loading data from CSV files.
    """
    
    def __init__(self, file_path):
        """
        Initializes the DataLoader with the file path to the CSV file.

        :param file_path: str, path to the CSV file
        """
        self.file_path = file_path

    def load_data(self):
        """
        Loads the CSV file into a pandas DataFrame.

        :return: pd.DataFrame containing the data from the CSV file
        """
        try:
            data = pd.read_csv(self.file_path)
            logger.info(f"Data successfully loaded from {self.file_path}")
            logger.info(f"DataFrame Shape: {data.shape}")
            return data
        except FileNotFoundError:
            logger.error(f"Error: File not found at {self.file_path}")
            return None
        except pd.errors.EmptyDataError:
            logger.error(f"Error: No data in file at {self.file_path}")
            return None
        except Exception as e:
            logger.error(f"An error occurred while loading the file: {e}")
            return None
# Example usage:
# loader = DataLoader("path_to_your_file.csv")
# df = loader.load_data()


# Load environment variables
load_dotenv()

# Fetch database connection parameters
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

class LoadSqlData:
    """
    A class to load SQL data from PostgreSQL using psycopg2 or SQLAlchemy.
    """
    
    def __init__(self, query):
        self.query = query

    def load_data_from_postgres(self):
        """
        Load data from PostgreSQL using psycopg2.

     
        """
        try:
            # Establish connection to the database
            connection = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD
            )

            
            # Fetch data using pandas
            df = pd.read_sql_query(self.query, connection)
            
            # Close connection
            connection.close()
            print('Sucessfully Loaded')
            return df
        except Exception as e:
            print(f"An error occurred while loading data with psycopg2: {e}")
            print("Connection parameters:")
            print(f"Host: {DB_HOST}, Port: {DB_PORT}, DB: {DB_NAME}, User: {DB_USER}, Password: {DB_PASSWORD}")
            return None


    def load_data_using_sqlalchemy(self):
        """
        Load data from PostgreSQL using SQLAlchemy.
        """
        try:
            # Create connection string
            connection_string = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
            
            # Create engine
            engine = create_engine(connection_string)

            # Fetch data using pandas
            df = pd.read_sql_query(self.query, engine)
            print('Sucessfully Loaded')
            return df
        except Exception as e:
            print(f"An error occurred while loading data with SQLAlchemy: {e}")
            return None


# Data clean and transformation

import logging
import pandas as pd
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self, dataframe):
        """
        Initialize the DataCleaner with a DataFrame.
        :param dataframe: pandas DataFrame to be cleaned.
        """
        self.df = dataframe

    def check_missing_values(self):
        """
        Check for missing values in each column.
        :return: DataFrame with columns, missing value count, and percentage.
        """
        logger.info("Checking for missing values in the DataFrame.")
        missing_info = self.df.isnull().sum()
        missing_percentage = (missing_info / len(self.df)) * 100
        logger.info("Missing values check completed.")
        return pd.DataFrame({
            'Column': self.df.columns,
            'Missing Values': missing_info,
            'Missing Percentage': missing_percentage,
            'Data Type': self.df.dtypes
        }).reset_index(drop=True)

    def remove_outliers(self, threshold=3):
        """
        Remove outliers from numerical columns based on Z-score.
        :param threshold: Z-score threshold for identifying outliers.
        """
        logger.info(f"Removing outliers with Z-score threshold of {threshold}.")
        try:
            numeric_cols = self.df.select_dtypes(include=['number'])
            z_scores = numeric_cols.apply(zscore)
            self.df = self.df[(np.abs(z_scores) < threshold).all(axis=1)]
            logger.info("Outlier removal completed.")
        except Exception as e:
            logger.error(f"An error occurred during outlier removal: {e}")

    def detect_outliers_plot(self, column):
        """
        Plot a boxplot to detect outliers in a numerical column.
        :param column: Name of the numerical column to plot.
        """
        logger.info(f"Generating outlier detection plot for column: {column}.")
        if column not in self.df.columns:
            logger.error(f"Column '{column}' does not exist in the DataFrame.")
            return
        if not np.issubdtype(self.df[column].dtype, np.number):
            logger.error(f"Column '{column}' is not numerical.")
            return
        try:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=self.df[column])
            plt.title(f"Outlier Detection in {column}")
            plt.show()
            logger.info(f"Outlier detection plot for '{column}' generated successfully.")
        except Exception as e:
            logger.error(f"An error occurred while plotting outliers: {e}")

    def transform_datetime(self, column, timezone):
        """
        Convert a datetime column to a specified timezone.
        :param column: Name of the datetime column.
        :param timezone: Target timezone (e.g., 'UTC', 'America/New_York').
        """
        logger.info(f"Transforming datetime column '{column}' to timezone '{timezone}'.")
        if column not in self.df.columns:
            logger.error(f"Column '{column}' does not exist in the DataFrame.")
            return
        try:
            self.df[column] = pd.to_datetime(self.df[column])
            self.df[column] = self.df[column].dt.tz_localize(None).dt.tz_localize(timezone)
            logger.info(f"Datetime transformation for column '{column}' completed.")
        except Exception as e:
            logger.error(f"An error occurred while transforming datetime: {e}")

    def fill_missing_values(self, strategy='mean', columns=None):
        """
        Fill missing values in specified columns.
        :param strategy: Strategy to fill missing values ('mean', 'median', 'mode').
        :param columns: List of columns to fill missing values. If None, all columns are processed.
        """
        logger.info(f"Filling missing values using strategy '{strategy}'.")
        if columns is None:
            columns = self.df.columns

        for column in columns:
            try:
                if self.df[column].dtype == 'object':
                    self.df[column].fillna(self.df[column].mode()[0], inplace=True)
                else:
                    if strategy == 'mean':
                        self.df[column].fillna(self.df[column].mean(), inplace=True)
                    elif strategy == 'median':
                        self.df[column].fillna(self.df[column].median(), inplace=True)
                    elif strategy == 'mode':
                        self.df[column].fillna(self.df[column].mode()[0], inplace=True)
                logger.info(f"Missing values in column '{column}' filled successfully.")
            except Exception as e:
                logger.error(f"An error occurred while filling missing values for column '{column}': {e}")

    def drop_column(self, column):
        """
        Drop a specified column from the DataFrame.
        :param column: Name of the column to drop.
        """
        logger.info(f"Dropping column: {column}.")
        if column not in self.df.columns:
            logger.error(f"Column '{column}' does not exist in the DataFrame.")
            return
        self.df.drop(columns=[column], inplace=True)
        logger.info(f"Column '{column}' dropped successfully.")

    def standardize_column_names(self):
        """
        Standardize column names by converting them to lowercase and replacing spaces with underscores.
        """
        logger.info("Standardizing column names.")
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_')
        logger.info("Column names standardized successfully.")

    def remove_duplicates(self):
        """
        Remove duplicate rows from the DataFrame.
        """
        logger.info("Removing duplicate rows from the DataFrame.")
        self.df.drop_duplicates(inplace=True)
        logger.info("Duplicate rows removed successfully.")

    def remove_nulls_from_columns(self, columns):
        """
        Remove rows with null values in the specified columns from the DataFrame.
        :param columns: List of column names to check for null values.
        """
        logger.info(f"Removing rows with null values in columns: {columns}.")
        if not isinstance(columns, list):
            logger.error("Columns parameter should be a list of column names.")
            return
        
        missing_columns = [col for col in columns if col not in self.df.columns]
        if missing_columns:
            logger.error(f"The following columns do not exist in the DataFrame: {', '.join(missing_columns)}")
            return

        self.df.dropna(subset=columns, inplace=True)
        logger.info(f"Rows with null values in columns {columns} removed successfully.")

    def get_cleaned_data(self):
        """
        Retrieve the cleaned DataFrame.
        :return: Cleaned DataFrame.
        """
        logger.info("Retrieving the cleaned DataFrame.")
        return self.df


 
'''

    # Initialize the cleaner
    cleaner = DataCleaner(df)

    # Check missing values
    print("Missing Values:")
    print(cleaner.check_missing_values())

    # Fill missing values
    cleaner.fill_missing_values()

    # Remove duplicates
    cleaner.remove_duplicates()

    # Standardize column names
    cleaner.standardize_column_names()

    # Transform datetime column
    cleaner.transform_datetime('joining_date', 'UTC')

    # Detect outliers plot
    print("\nOutlier Detection Plot:")
    cleaner.detect_outliers_plot('salary')

    # Remove outliers
    cleaner.remove_outliers()

    # Drop a column
    cleaner.drop_column('name')

    # Get cleaned data
    cleaned_data = cleaner.get_cleaned_data()
    print("\nCleaned DataFrame:")
    print(cleaned_data)
'''


