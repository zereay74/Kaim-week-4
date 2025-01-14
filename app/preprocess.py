import pandas as pd
import logging
import sys
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_csv(self):  

        data_frame = pd.read_csv(self.file_path)
        logger.info(f" Dataframe loaded  sucessfully with shape {data_frame.shape}")
        return data_frame
    
    def merge_data(self, data_frame1, data_frame2, how, on):
        # Merging DataFrames
        merged_df = data_frame1.merge(data_frame2, how=how, on=on)
        logger.info(f"Merged DataFrame shape: {merged_df.shape}")
        return merged_df
    
    def drop_columns(self, data_frame, drop_columns):
        # Dropping columns
        data_frame.drop(columns=drop_columns, inplace=True)
        logger.info(f"Columns dropped: {drop_columns}")
        logger.info(f"DataFrame shape after dropping columns: {data_frame.shape}")
        return data_frame

    def replace_zero(self, data_frame, column):
        data_frame[column] = data_frame[column].apply(lambda x: 0 if str(x) == '0' else x)
        logger.info(f"Replaced zero-like values with integer 0 in column '{column}'.")
        return data_frame
    
    def remove_rows_with_store_zero(self, data_frame, column):
        """
        When the store is closed, the sale is zero. So rows with sales 0 doesn't make sense.
        But it's more efficient to remove rows with store 0. 
        """
        data_frame = data_frame[data_frame[column] ==1].copy()
        logger.info(f"Removed rows with store 0")
        return data_frame   

    def extract_date(self, data_frame, date_column):
        # Convert 'Date' column to datetime format
        data_frame[date_column] = pd.to_datetime(data_frame[date_column])

        # Extract Year, Month, and Day
        data_frame['Year'] = data_frame[date_column].dt.year
        data_frame['Month'] = data_frame[date_column].dt.month
        data_frame['Day'] = data_frame[date_column].dt.day

        logger.info(f"Extracted Year, Month, and Day from '{date_column}' column.")
        return data_frame
    
    def group_and_plot(self,data_frame, group_by_columns=['Year', 'Month'],  plot_column='Sales'):
      
        # Grouping by year and month, and summing up the sales for each group
        grouped_data = data_frame.groupby(group_by_columns)[plot_column].sum().reset_index()

        # Generating x-axis labels (e.g., "Year-Month")
        grouped_data['Year-Month'] = grouped_data['Year'].astype(str) + '-' + grouped_data['Month'].astype(str)

        # Plotting the data
        plt.figure(figsize=(12, 6))
        plt.plot(grouped_data['Year-Month'], grouped_data[plot_column], marker='o')
        plt.title(f'{plot_column} Over Time')
        plt.xlabel('Year-Month')
        plt.ylabel(plot_column)
        plt.xticks(rotation=45)  # Rotating x-axis labels for better readability
        plt.grid(True)
        plt.tight_layout()
        logger.info("Plotting Monthly Sales Over Time.")
        plt.show()
        
    ''' 
    # extract date for test dataframe
    def limit_date (self, train_dataframe, val_dataframe, column, min_date, max_date):
        train_dataframe = train_dataframe[train_dataframe[column].dt.year <= min_date] 
        val_dataframe = val_dataframe[val_dataframe[column].dt.year == max_date]
        logger.info(f"Limiting date range and splitting dataframe to train and validation dataframes")
        return train_dataframe, val_dataframe
    '''
