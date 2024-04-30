import pandas as pd
from django.shortcuts import render
from django.conf import settings
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
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


# utils.py

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train_logistic_regression(dataset, target_column_name, test_size=0.2, random_state=42):
    # Separate features (X) and target variable (y)
    X = dataset.drop(columns=target_column_name)
    y = dataset[target_column_name]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initialize the Logistic Regression classifier
    log_reg_classifier = LogisticRegression()

    # Train the classifier on the training data
    log_reg_classifier.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = log_reg_classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_decision_tree(dataset, target_column_name, test_size=0.2, random_state=42):
    # Separate features (X) and target variable (y)
    X = dataset.drop(columns=target_column_name)
    y = dataset[target_column_name]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initialize the Decision Tree classifier
    decision_tree_classifier = DecisionTreeClassifier()

    # Train the classifier on the training data
    decision_tree_classifier.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = decision_tree_classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier

def train_random_forest(dataset, target_column_name, test_size=0.2, random_state=42):
    # Separate features (X) and target variable (y)
    X = dataset.drop(columns=target_column_name)
    y = dataset[target_column_name]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initialize the Random Forest classifier
    random_forest_classifier = RandomForestClassifier()

    # Train the classifier on the training data
    random_forest_classifier.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = random_forest_classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def train_gradient_boosting(dataset, target_column_name, test_size=0.2, random_state=42):
    # Separate features (X) and target variable (y)
    X = dataset.drop(columns=target_column_name)
    y = dataset[target_column_name]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initialize the Gradient Boosting Machine regressor
    gbm_regressor = GradientBoostingRegressor()

    # Train the regressor on the training data
    gbm_regressor.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = gbm_regressor.predict(X_test)

    # Calculate accuracy
    accuracy = r2_score(y_test, y_pred)  # Since it's regression, we use R-squared score as accuracy
    return accuracy


from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

def train_svm(dataset, target_column_name, test_size=0.2, random_state=42):
    # Separate features (X) and target variable (y)
    X = dataset.drop(columns=target_column_name)
    y = dataset[target_column_name]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initialize the SVM classifier
    svm_classifier = SVC()

    # Train the classifier on the training data
    svm_classifier.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = svm_classifier.predict(X_test)

    # Calculate accuracy using F1 score
    accuracy = f1_score(y_test, y_pred, average='weighted')  # Using weighted average F1 score for multiclass classification
    return accuracy


from sklearn.neighbors import KNeighborsClassifier

def train_knn(dataset, target_column_name, test_size=0.2, random_state=42):
    # Separate features (X) and target variable (y)
    X = dataset.drop(columns=target_column_name)
    y = dataset[target_column_name]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initialize the KNN classifier
    knn_classifier = KNeighborsClassifier()

    # Train the classifier on the training data
    knn_classifier.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = knn_classifier.predict(X_test)

    # Calculate accuracy using F1 score
    accuracy = f1_score(y_test, y_pred, average='weighted')  # Using weighted average F1 score for multiclass classification
    return accuracy


from sklearn.naive_bayes import GaussianNB

def train_naive_bayes(dataset, target_column_name, test_size=0.2, random_state=42):
    # Separate features (X) and target variable (y)
    X = dataset.drop(columns=target_column_name)
    y = dataset[target_column_name]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initialize the Naive Bayes classifier
    naive_bayes_classifier = GaussianNB()

    # Train the classifier on the training data
    naive_bayes_classifier.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = naive_bayes_classifier.predict(X_test)

    # Calculate accuracy using F1 score
    accuracy = f1_score(y_test, y_pred, average='weighted')  # Using weighted average F1 score for multiclass classification
    return accuracy


from sklearn.cluster import KMeans

def train_kmeans(dataset, num_clusters=12, random_state=42):
    # Initialize the KMeans clustering model
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)

    # Fit the model to the dataset
    kmeans.fit(dataset)

    # Assign clusters to data points
    clusters = kmeans.predict(dataset)

    return clusters


from sklearn.cluster import AgglomerativeClustering

def train_hierarchical_clustering(dataset, num_clusters=2):
    # Initialize the Agglomerative Clustering model
    hierarchical_clustering = AgglomerativeClustering(n_clusters=num_clusters)

    # Fit the model to the dataset
    clusters = hierarchical_clustering.fit_predict(dataset)

    return clusters
