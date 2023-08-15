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
