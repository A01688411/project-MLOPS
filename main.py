"""
This is a script for predicting rain using an artificial neural network. 
The process includes loading data, visualizing the data, cleaning the data,
preprocessing the data, building the model.

"""
# Import necessary libraries
#import pandas as pd
import numpy as np  # For linear algebra
from load.load_data import load_and_examine_data
from load.load_data import plot_count_and_correlation
from load.load_data import create_cyclic_features
from load.load_data import fill_missing_values
from load.load_data import visualize_over_years
from preprocess.preprocess_data import encode_categorical_features
from preprocess.preprocess_data import  scale_features
from preprocess.preprocess_data import  remove_outliers
from train.train_data import Preprocessor
from train.train_data import ModelBuilder
from train.train_data import Evaluator

#from typing import List, Dict, Tuple
#from sklearn.preprocessing import LabelEncoder, StandardScaler

# Prepare a random seed for reproducibility
np.random.seed(0)


# LOADING DATA

# Utilize the function
data = load_and_examine_data("./data/weatherAUS.csv")


# DATA VISUALIZATION AND CLEANING

plot_count_and_correlation(data, "RainTomorrow", ["#C2C4E2","#EED4E5"])
data = create_cyclic_features(data)
data = fill_missing_values(data)
visualize_over_years(data)


## DATA PREPROCESSING

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