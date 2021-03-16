'''
Jason Liu, Yoonseo Song, Daniel Miau
CSE 163
Final Project
This program uses ML models to analyze the Cleveland heart disease data
set from the UCI Machine Learning Repository
'''

import pandas as pd
import numpy as np
import models


def main():
    # Reads in the file
    data = pd.read_csv('https://raw.githubusercontent.com/yoonseosong/cse163-'
                       'final-project/master/processed.cleveland.csv')
    # FOR TESTING: checks the shape of the dataframe before dropping rows
    # with empty values
    # print(data.shape)

    # Drops '?' data points
    data = data.replace('?', np.NaN)
    data = data.dropna()
    # FOR TESTING: checks the shape of the dataframe after dropping rows
    # with empty values
    # print(data.shape)

    # Stores results
    all_results = {}

    # Generates list of features
    features = data.copy()
    features = features.loc[:, data.columns != 'num']
    features_list = features.columns

    # Sets how many times to run each model
    iterations = 100

    # Create model using all features
    models.create_model(None, data, all_results, iterations)

    # Create models with 1 dropped feature each
    for feature in features_list:
        models.create_model(feature, data, all_results, iterations)

    # Converts results to a pandas dataframe
    all_results = pd.DataFrame.from_dict(all_results)
    # FOR TESTING: checks to make sure all_results is a pandas dataframe
    # and checks the results
    # print(type(all_results))
    # print(all_results)

    # Creates a boxplot of the results and finds the p-value through ANOVA
    models.analyze_results(all_results)


if __name__ == '__main__':
    main()