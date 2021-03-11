# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# Hello. It's Daniel

# C:\\Users\\Jason Liu\\PycharmProjects\\cse163-final-project\\processed.cleveland.csv

# Hello. It's Jason.

import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib
import scipy
import numpy as np


def main():
    # Reads in the file
    data = pd.read_csv('C:\\Users\\Jason Liu\\PycharmProjects\\cse163-final-project\\processed.cleveland.csv')

    # Drops '?' datapoints
    data = data.replace('?', np.NaN)
    data = data.dropna()

    # Features
    features = data.copy()
    features = features.loc[:, data.columns != 'num']
    features_list = features.columns
    categorical_variables = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
    features = pd.get_dummies(features, columns=categorical_variables)

    # Labels
    labels = data['num']

    # Stores all score values
    all_results = {}

    # Model with all features
    all_results['all'] = []
    for i in range(10):
        clf = RandomForestClassifier(n_estimators=10, random_state=1)
        features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3)
        clf = clf.fit(features_train, labels_train)
        test_score = clf.score(features_test, labels_test)
        all_results['all'].append(test_score)


def create_model(dropped_column, data, results, iterations):
    features, labels = process_data(data, dropped_column)
    if dropped_column is None:
        dropped_column = 'all'
    if dropped_column not in results:
        results[dropped_column] = []
    for i in range(iterations):
        clf = RandomForestClassifier(n_estimators=10)
        features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3)
        clf = clf.fit(features_train, labels_train)
        test_score = clf.score(features_test, labels_test)
        results[dropped_column].append(test_score)


def process_data(data, dropped_column):
    features = data.copy()
    features = features.loc[:, data.columns != 'num']
    features_list = features.columns
    categorical_variables = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
    if dropped_column is not None:
        features_list.remove(dropped_column)
        if dropped_column in categorical_variables:
            categorical_variables.remove(dropped_column)
    features = pd.get_dummies(features, columns=categorical_variables)
    labels = data['num']
    return features, labels


if __name__ == '__main__':
    main()
