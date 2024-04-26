import pandas as pd
from django.shortcuts import render
from django.conf import settings
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def auto_preprocess_dataset(dataset):
    # Remove null values
    dataset = dataset.dropna()

    # Separate categorical and numerical columns
    categorical_cols = dataset.select_dtypes(include=['object']).columns
    numerical_cols = dataset.select_dtypes(include=['number']).columns

    # Apply label encoding to categorical columns
    if len(categorical_cols) > 0:
        encoder = LabelEncoder()
        for col in categorical_cols:
            dataset[col] = encoder.fit_transform(dataset[col])

    # Apply min-max scaling to numerical columns
    if len(numerical_cols) > 0:
        scaler = MinMaxScaler()
        dataset[numerical_cols] = scaler.fit_transform(dataset[numerical_cols])

    return dataset
