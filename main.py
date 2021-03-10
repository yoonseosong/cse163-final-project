# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# Hello. It's Daniel

# C:\\Users\\Jason Liu\\PycharmProjects\\cse163-final-project\\processed.cleveland.csv

#Hello. It's Jason.

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
    data = pd.read_csv('C:\\Users\\Jason Liu\\PycharmProjects\\cse163-final-project\\processed.cleveland.csv')

    data = data.replace('?', np.NaN)
    data = data.dropna()

    all_features = data.copy()
    all_features = all_features.loc[:, data.columns != 'num']
    all_features = pd.get_dummies(all_features, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal'])
    #print(all_features)

    labels = data['num']

    clf = RandomForestClassifier(n_estimators=5)

    features_train, features_test, labels_train, labels_test = train_test_split(all_features, labels, test_size=0.3)

    clf = clf.fit(features_train, labels_train)

    test_score = clf.score(features_test, labels_test)
    #print(test_score)

    scores = cross_val_score(clf, features_test, labels_test, cv=2)


    model = DecisionTreeClassifier()
    model.fit(features_train, labels_train)
    label_predictions = model.predict(features_test)
    train_acc = accuracy_score(labels_test, label_predictions)
    print('Accuracy:', train_acc)

    #print(scores.mean())




















if __name__ == '__main__':
    main()
