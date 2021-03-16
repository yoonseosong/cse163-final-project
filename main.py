'''
DocString
'''

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
from matplotlib import pyplot as plt
import scipy.stats as stats
import numpy as np


def main():
    # Reads in the file
    data = pd.read_csv('https://raw.githubusercontent.com/yoonseosong/cse163-'
                       'final-project/master/processed.cleveland.csv')

    # Drops '?' data points
    data = data.replace('?', np.NaN)
    data = data.dropna()

    # Stores results
    all_results = {}

    # Generates list of features
    features = data.copy()
    features = features.loc[:, data.columns != 'num']
    features_list = features.columns

    # Sets how many times to run each model
    iterations = 100

    # Create model using all features
    create_model(None, data, all_results, iterations)

    # Create models with 1 dropped feature each
    for feature in features_list:
        create_model(feature, data, all_results, iterations)

    # Converts results to a pandas dataframe
    all_results = pd.DataFrame.from_dict(all_results)
    #print(all_results.columns)
    #print(all_results)

    # Creates a boxplot of the results and finds the p-value through ANOVA
    analyze_results(all_results)


def create_model(dropped_column, data, results, iterations):
    '''
    Creates a ML model using the inputted data while dropping
    the specified column
    Creates the specified number (iterations) of models and stores
    the results in results
    '''
    features, labels = process_data(data, dropped_column)
    if dropped_column is None:
        dropped_column = 'none'
    if dropped_column not in results:
        results[dropped_column] = []
    for i in range(iterations):
        '''
        Uncomment 1 classifier at a time
        '''
        clf = RandomForestClassifier(n_estimators=10)
        # clf = KNeighborsClassifier()
        # clf = MLPClassifier()
        # clf = AdaBoostClassifier()
        # clf = DecisionTreeClassifier()
        features_train, features_test, labels_train, labels_test =\
            train_test_split(features, labels, test_size=0.3)
        clf = clf.fit(features_train, labels_train)
        test_score = clf.score(features_test, labels_test)
        results[dropped_column].append(test_score)


def process_data(data, dropped_column):
    '''
    Takes in data and removes dropped_column from the data
    If dropped_column is None, no columns are dropped
    Creates dummy variables for categorical variables
    Returns a tuple with features and labels
    '''
    features = data.copy()
    features = features.loc[:, data.columns != 'num']
    features_list = list(features.columns)
    categorical_variables = ['sex', 'cp', 'fbs', 'restecg', 'exang',
                             'slope', 'thal']
    if dropped_column is not None:
        features_list.remove(dropped_column)
        if dropped_column in categorical_variables:
            categorical_variables.remove(dropped_column)
    features = pd.get_dummies(features, columns=categorical_variables)
    labels = data['num']
    return features, labels


def analyze_results(all_results):
    '''
    Takes in the results from our ML models and runs an ANOVA test
    Plots the results as a boxplot along with the ANOVA p value
    Creates a table with means and standard deviations
    '''

    # ANOVA
    f_value, p_value = stats.f_oneway(all_results['none'], all_results['age'],
                                      all_results['sex'], all_results['cp'],
                                      all_results['trestbps'],
                                      all_results['chol'], all_results['fbs'],
                                      all_results['restecg'],
                                      all_results['thalach'],
                                      all_results['exang'],
                                      all_results['oldpeak'],
                                      all_results['slope'],
                                      all_results['ca'],
                                      all_results['thal'])

    # Plot a boxplot of model accuracies
    results_melt = pd.melt(all_results.reset_index(), id_vars=['index'], value_vars=list(all_results.columns))
    results_melt.columns = ['index', 'feature', 'score']
    ax = sns.boxplot(x='feature', y='score', data=results_melt)
    plt.title('Model Accuracy with Removed Features  p=' + str(round(p_value, 3)))
    plt.xlabel('Removed Feature')
    plt.ylabel('Accuracy Score')
    plt.xticks(rotation=-30)
    plt.tight_layout()

    # Creates a table for the accuracy scores with the mean and standard deviations
    row_name = ['mean', 'sd']
    cell_text = [[], []]
    for column in all_results.columns:
        curr_column = all_results[[column]].transpose()
        curr_column = curr_column.values.tolist()[0]
        cell_text[0].append(round(sum(curr_column) / len(curr_column), 3))
        cell_text[1].append(round(np.std(curr_column), 3))
    fig, ax1 = plt.subplots(1, 1, figsize=(15, 8))
    ax1.xaxis.set_label_position('top')
    ax1.set_title('Means and Standard Deviations of Accuracy Scores')
    # plt.title('Means and Standard Deviations of Accuracy Scores')
    ax1.axis('tight')
    ax1.axis('off')
    ax1.table(cellText=cell_text, colLabels=all_results.columns, rowLabels=row_name, loc="center")

    plt.show()


if __name__ == '__main__':
    main()
