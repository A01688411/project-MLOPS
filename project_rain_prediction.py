"""
This is a script for predicting rain using an artificial neural network. 
The process includes loading data, visualizing the data, cleaning the data,
preprocessing the data, building the model, and finally drawing some conclusion.

The script uses the Kaggle Python Docker image, and the input data files
should be available in the read-only directory "../input/".
"""

# Import necessary libraries
import numpy as np  # For linear algebra
import pandas as pd  # For data processing
import os  # For operating system dependent functionalities

import matplotlib.pyplot as plt  # For data visualization
import seaborn as sns  # For data visualization
from sklearn import preprocessing  # For preprocessing data
from sklearn.preprocessing import StandardScaler  # For standardizing features
from sklearn.model_selection import train_test_split  # For splitting the data into training and test datasets

from keras.layers import Dense, BatchNormalization, Dropout, LSTM
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import Adam
from tensorflow.keras import regularizers  # For implementing regularization in the model
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score  # for evaluating model performance

# Prepare a random seed for reproducibility
np.random.seed(0)

def list_data_files(directory: str='/kaggle/input') -> None:
    """
    This function lists all the files under the input directory.
    
    Args:
    directory (str): A string representing the directory to check files from.
    """
    try:
        for dirname, _, filenames in os.walk(directory):
            for filename in filenames:
                print(os.path.join(dirname, filename))
                
    except Exception as e:
        print("An error occurred while trying to list the files: ", e)



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

# Utilize the function
data = load_and_examine_data("/content/weatherAUS.csv")



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



plot_count_and_correlation(data, "RainTomorrow", ["#C2C4E2","#EED4E5"])
data = create_cyclic_features(data)
data = fill_missing_values(data)
visualize_over_years(data)


## DATA PREPROCESSING

import pandas as pd
from typing import List, Dict, Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler

def encode_categorical_features(data: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
    """
    This function encodes categorical features to numeric using LabelEncoder.

    Args:
    data (pd.DataFrame): The DataFrame to encode.
    categorical_cols (List[str]): The list of categorical column names to encode.

    Returns:
    pd.DataFrame: The DataFrame with encoded categorical columns.
    """
    label_encoder = LabelEncoder()
    for col in categorical_cols:
        data[col] = label_encoder.fit_transform(data[col])
    return data


def scale_features(features: pd.DataFrame) -> pd.DataFrame:
    """
    This function scales features using StandardScaler.

    Args:
    features (pd.DataFrame): The DataFrame of features to scale.

    Returns:
    pd.DataFrame: The DataFrame with scaled features.
    """
    col_names = list(features.columns)
    s_scaler = StandardScaler()
    features = s_scaler.fit_transform(features)
    features = pd.DataFrame(features, columns=col_names)
    return features


def remove_outliers(features: pd.DataFrame, bounds: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """
    This function detects and removes outliers based on given bounds.

    Args:
    features (pd.DataFrame): The DataFrame to remove outliers from.
    bounds (Dict[str, Tuple[float, float]]): The outlier bounds for each feature.

    Returns:
    pd.DataFrame: The DataFrame without outliers.
    """
    for feature, (lower, upper) in bounds.items():
        features = features[(features[feature] > lower) & (features[feature] < upper)]
        return features


# Apply categorical feature encoding
categorical_cols = [col for col in data.columns if data[col].dtype == 'object']
data = encode_categorical_features(data, categorical_cols)

# Select features
features_cols = data.columns.difference(['RainTomorrow', 'Date','day', 'month'])
features = data[features_cols]
target = data["RainTomorrow"]

# Scale features
features = scale_features(features)

# Remove outliers
outlier_bounds = {
    "MinTemp": (-2.3, 2.3),
    "MaxTemp": (-2, 2.3),
    "Rainfall": (None, 4.5),
    "Evaporation": (None, 2.8),
    "Sunshine": (None, 2.1),
    "WindGustSpeed": (-4, 4),
    "WindSpeed9am": (None, 4),
    "WindSpeed3pm": (None, 2.5),
    "Humidity9am": (-3, None),
    "Humidity3pm": (-2.2, None),
    "Pressure9am": (-2.7, 2),
    "Pressure3pm": (-2.7, 2),
    "Cloud9am": (None, 1.8),
    "Cloud3pm": (None, 2),
    "Temp9am": (-2, 2.3),
    "Temp3pm": (-2, 2.3)
}

features = remove_outliers(features, outlier_bounds)

# Assign target variable
features["RainTomorrow"] = target

## MODEL BUILDING

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks

class Preprocessor:
    """Custom transformer for data preprocessing tasks"""

    def __init__(self):
        pass
      
    def transform(self, data):
        """Preprocess the data"""
        X = data.drop(["RainTomorrow"], axis=1)
        #X = data.drop(["RainTomorrow"])
        y = data["RainTomorrow"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

class ModelBuilder:
    """Build and train the neural network model"""

    def __init__(self):
        pass

    def build_model(self, input_dim):
        """Build the neural network model"""
        model = Sequential()
        model.add(Dense(units=32, kernel_initializer='uniform', activation='relu', input_dim=input_dim))
        model.add(Dense(units=32, kernel_initializer='uniform', activation='relu'))
        model.add(Dense(units=16, kernel_initializer='uniform', activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(units=8, kernel_initializer='uniform', activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
        return model

    def train_model(self, model, X_train, y_train):
        """Train the neural network model"""
        early_stopping = callbacks.EarlyStopping(
            min_delta=0.001,
            patience=20,
            restore_best_weights=True,
        )
        opt = Adam(learning_rate=0.00009)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(X_train, y_train, batch_size=32, epochs=150, callbacks=[early_stopping], validation_split=0.2)
        return history

class Evaluator:
    """Evaluate the trained model"""

    def __init__(self):
        pass

    def plot_loss(self, history):
        """Plot training and validation loss over epochs"""
        history_df = pd.DataFrame(history.history)
        plt.plot(history_df['loss'], "#BDE2E2", label='Training loss')
        plt.plot(history_df['val_loss'], "#C2C4E2", label='Validation loss')
        plt.title('Training and Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc="best")
        plt.show()

    def plot_accuracy(self, history):
        """Plot training and validation accuracy over epochs"""
        history_df = pd.DataFrame(history.history)
        plt.plot(history_df['accuracy'], "#BDE2E2", label='Training accuracy')
        plt.plot(history_df['val_accuracy'], "#C2C4E2", label='Validation accuracy')
        plt.title('Training and Validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    def evaluate_model(self, model, X_test, y_test):
        """Evaluate the model on the test set"""
        y_pred = model.predict(X_test)
        y_pred = (y_pred > 0.5)
        cmap1 = sns.diverging_palette(260, -10, s=50, l=75, n=5, as_cmap=True)
        plt.subplots(figsize=(12, 8))
        cf_matrix = confusion_matrix(y_test, y_pred)
        sns.heatmap(cf_matrix / np.sum(cf_matrix), cmap=cmap1, annot=True, annot_kws={'size': 15})
        print(classification_report(y_test, y_pred))

class Main:
    """Main function to coordinate the pipeline"""

    def __init__(self):
        pass

    def run(self):
        """Execute the pipeline"""
        # Load the features data
        #features = pd.read_csv("/content/weatherAUS.csv")

        # Preprocess the data
        preprocessor = Preprocessor()
        X_train, X_test, y_train, y_test = preprocessor.transform(features)

        # Build the model
        model_builder = ModelBuilder()
        model = model_builder.build_model(input_dim=X_train.shape[1])

        # Train the model
        history = model_builder.train_model(model, X_train, y_train)

        # Plot loss and accuracy
        evaluator = Evaluator()
        evaluator.plot_loss(history)
        evaluator.plot_accuracy(history)

        # Evaluate the model
        evaluator.evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main = Main()
    main.run()