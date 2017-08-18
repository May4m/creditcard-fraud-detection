
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


def oversample_dataset(dataset, oversample_factor=10):
    """
    applies oversampling by duplicating the minority dataset
    """

    dataset = dataset.append(n_duplicates * [dataset[dataset['Class'] == 1]], ignore_index=True)  # oversampling by duplication
    return 


def undersample_dataset(dataset):
    """
    Applies undersampling by selecting the number of points in the majority equal to the length of
    the minority.
    """

    num_of_fraud_points = len(dataset[dataset['Class'] == 1])
    fraud_indices = np.array(dataset[dataset['Class'] == 1].index)
    non_fraud_indices = np.array(dataset[dataset['Class'] == 0].index)

    # randomly select (n=number of fraud sample) samples from the non-fraud subset -> balanced dataset
    random_non_fraud_indices = np.random.choice(non_fraud_indices, num_of_fraud_points, replace=False)
    total_undersampled_dataset_indices = np.concatenate([random_non_fraud_indices, fraud_indices])
    undersampled_dataset = dataset.iloc[total_undersampled_dataset_indices, :]
    return undersampled_dataset


def load_dataset(filename="creditcard.csv", dataset_split_method='train-test',
                sampling_method='undersampling', train_test_ratio=None, kfolds=None):
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

    # apply sampling technique if selected to tackling class imbalance
    if sampling_method == 'oversampling':
        dataset = oversample_dataset(dataset)
    elif sampling_method == 'undersampling':
        dataset = undersample_dataset(dataset)

    features = dataset.ix[:, dataset.columns != 'Class']
    classes = dataset['Class']

    # train/test split method
    if dataset_split_method == 'train-test':
        X_train, X_test, y_train, y_test = train_test_split(features, classes, test_size=train_test_ratio, random_state=0)
        # remove duplicates in test for accuracy metrics. only when oversampling is applied
        if sampling_method == 'oversampling':
            X_test = X_test.drop_duplicates()
            y_test = y_test[X_test.index]
        return {'x-train': X_train, 'y-train': y_train, 'x-test': X_test, 'y-test': y_test}, dataset_split_method

    # kfolds method
    if dataset_split_method == 'kfolds':
        kfolds = KFold(len(classes), kfolds, shuffle=False) 
    return {'features': features, 'classes': classes, 'kfolds': kfolds}, dataset_split_method


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



def model(datainput, split_method):
    """
    random forest model
    """

    # random forest classifiers
    if split_method == "train-test":
        dataset = datainput
        ran_forest = RandomForestClassifier(n_estimators=13, n_jobs=-1)
        ran_forest.fit(dataset['x-train'], dataset['y-train'])
        accaracy_measures(ran_forest, dataset)

model(*load_dataset())