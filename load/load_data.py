
# Import necessary libraries
import numpy as np  # For linear algebra
import pandas as pd  # For data processing
import os  # For operating system dependent functionalities

import matplotlib.pyplot as plt  # For data visualization
import seaborn as sns  # For data visualization
from sklearn import preprocessing  # For preprocessing data
from sklearn.preprocessing import StandardScaler  # For standardizing features
from sklearn.model_selection import train_test_split  # For splitting the data into training and test datasets


# LOADING DATA
def load_and_examine_data(filepath: str) -> pd.DataFrame:
    """
    This function loads data from a CSV file, displays the first few records 
    and prints information about the data.
    
    Args:
    filepath (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded DataFrame.
    """
    try:
        # Load the data
        data = pd.read_csv(filepath)

        # Print first few records
        print("First few records:")
        print(data.head())
        
        # Print data info
        print("\nData Information:")
        data.info()
        
        return data
    except FileNotFoundError:
        print(f"{filepath} does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# DATA VISUALIZATION AND CLEANING
from typing import List

def plot_count_and_correlation(data: pd.DataFrame, target_col: str, color_palette: List[str]) -> None:
    """
    This function plots a countplot for the target column, and a correlation heatmap for numeric columns.
    """
    # Count plot of target column
    sns.countplot(x= data[target_col], palette=color_palette)
    ...
    # Correlation amongst numeric attributes
    corrmat = data.corr()
    cmap = sns.diverging_palette(260,-10,s=50, l=75, n=6, as_cmap=True)
    plt.subplots(figsize=(18,18))
    sns.heatmap(corrmat, cmap=cmap,annot=True, square=True)


def encode(data: pd.DataFrame, col: str, max_val: int) -> pd.DataFrame:
    """
    This function encodes a column of a DataFrame as cyclic features using sine and cosine transformations.
    
    Args:
    data (pd.DataFrame): The DataFrame to encode.
    col (str): The column in the DataFrame to encode.
    max_val (int): The maximum value of the column, used for normalization.

    Returns:
    pd.DataFrame: The DataFrame with the encoded column.
    """
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data


def create_cyclic_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    This function parses dates into datetime and create cyclic features (sine/cosine) for days and months.
    """
    # Parsing datetime
    data['Date']= pd.to_datetime(data["Date"])
    # Creating a collumn of year
    data['year'] = data.Date.dt.year
    # Encoding months and days as cyclic features
    data['month'] = data.Date.dt.month
    data= encode(data, 'month', 12)
    data['day'] = data.Date.dt.day
    data = encode(data, 'day', 31)
    return data

def fill_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    This function fills missing values for both categorical and numeric variables. 
    For categorical variables, it fills with mode. 
    For numeric variables, it fills with median.
    """
    # Filling missing values in categorical variables
    object_cols = [col for col in data.columns if data[col].dtype == 'object']
    for col in object_cols:
        data[col].fillna(data[col].mode()[0], inplace=True)
    # Filling missing values in numeric variables
    num_cols = [col for col in data.columns if data[col].dtype == 'float64']
    for col in num_cols:
        data[col].fillna(data[col].median(), inplace=True)
    return data

def visualize_over_years(data: pd.DataFrame) -> None:
    """
    This function visualizes Rainfall and WindGustSpeed over years.
    Date column required.
    """
    plt.figure(figsize=(12,8))
    sns.lineplot(x=data['Date'].dt.year, y="Rainfall", data=data, color="#C2C4E2")
    plt.figure(figsize=(12,8))
    sns.barplot(x=data['Date'].dt.year, y="WindGustSpeed", data=data, ci =None, palette = ["#D0DBEE", "#C2C4E2", "#EED4E5", "#D1E6DC", "#BDE2E2"])

