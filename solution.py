
import math

import pandas as pd
import numpy as np

from sklearn.decomposition import KernelPCA, PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler



def load_dataset(filename="creditcard.csv", n_duplicates=3, split_ratio=0.2):
    """
    interface function to load the dataset and perform cross-validation splitting
    """
    h5 = False
    # HDFS is a faster when reading
    try:
        dataset = pd.read_hdf('creditcard.h5', 'creditcard')
        h5 = True
    except:
        h5 = False
        dataset = pd.read_csv('creditcard.csv')

    # try saving to hdfs
    if h5 is False:
        try:
            store = pd.HDFStore('creditcard.h5')
            store['creditcard'] = dataset
        except:
            print("[ WARNING ] could not save to HDFS format")

    # preprocessing
    dataset['norm_amount'] = StandardScaler().fit_transform(dataset['Amount'].values.reshape(-1, 1))
    dataset = dataset.drop(['Time','Amount'], axis=1)
    dataset = dataset.append(n_duplicates * [dataset[dataset['Class'] == 1]], ignore_index=True)  # oversampling by duplication



    
    features = dataset.ix[:, dataset.columns != 'Class']
    classes = dataset['Class']

    zero = dataset[classes == 0]

    # cross-validation
    X_train, X_test, y_train, y_test = train_test_split(features, classes, test_size=split_ratio, random_state=0)
    # remove duplicates in test for accuracy metrics
    X_test = X_test.drop_duplicates()
    y_test = y_test[X_test.index]


    return {'x-train': X_train, 'y-train': y_train, 'x-test': X_test, 'y-test': y_test, 'zero-class': zero}


def accaracy_measures(model, dataset):
    """
    prints common accurary measures of the model
    """

    predicted_y, true_y = model.predict(dataset['x-test']), dataset['y-test']

    print "****************%s***************" % str(model).rpartition('(')[0]

    print "score: ", model.score(
                        dataset['x-test'],
                        dataset['y-test'])
    
    # confusion matrix
    print "[Tn: %s] [Fp: %s] [Fn: %s] [Tp: %s]" % tuple(confusion_matrix(true_y, predicted_y).ravel())

    # precision recall
    print "precision: %s  recall: %s  f1-score: %s" % (precision_score(true_y, predicted_y), recall_score(true_y, predicted_y), f1_score(true_y, predicted_y))
    print "------------------------------------------------\n"



def model(dataset):
    """
    random forest model
    """

    # random forest classifiers
    ran_forest = RandomForestClassifier(n_estimators=13, n_jobs=-1)
    ran_forest.fit(dataset['x-train'], dataset['y-train'])


    accaracy_measures(ran_forest, dataset)


model(load_dataset(split_ratio=0.3))