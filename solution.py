
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


import visualization as viz


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
                sampling_method='undersampling', train_test_ratio=0.5, kfolds=5):
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

    X_train, X_test, y_train, y_test = train_test_split(features, classes, test_size=train_test_ratio, random_state=0)
    dataset_config = {'x-train': X_train, 'y-train': y_train, 'x-test': X_test, 'y-test': y_test}

    # kfolds method
    if dataset_split_method == 'kfolds':
        s = KFold(n_splits=2)
        kfolds = KFold(kfolds, shuffle=True)
        dataset_config['kfold'] = kfolds
        return dataset_config, 'kfolds'

    if sampling_method == 'oversampling':
        dataset_config['x-test'] = X_test.drop_duplicates()
        dataset_config['y-test'] = y_test[X_test.index]
    return dataset_config, 'train-test'


def accaracy_measures(model, points, conf_mat=False, roc_curve=False):
    x_plot_area = int(conf_mat) + int(roc_curve) + 1
    viz.plt.figure()

    # visualization confusion matrix
    if conf_mat:
        conf_matrix = confusion_matrix(points['y-test'], model.predict(points['x-test']))
        viz.plt.subplot(1, x_plot_area ,x_plot_area - 1)
        viz.plot_confusion_matrix(conf_matrix, classes=[0, 1], title='Confusion matrix')

    
    # visualize ROC curve
    if roc_curve:

        y_pred = model.predict(points['x-test'])
        viz.plt.subplot(1, x_plot_area, x_plot_area - 2)
        viz.plot_roc_curve(points['y-test'], y_pred)
    viz.plt.show()


def train_on_data(datainput, split_method, model=LogisticRegression(C=0.1, penalty='l1')):
    """
    random forest model
    """

    # TODO: use multiple modelds for training and evaluation

    print "training of ", model

    # when train-split validation method is applied
    if split_method == "train-test":
        model.fit(datainput['x-train'], datainput['y-train'])
        accaracy_measures(model, datainput, conf_mat=True)

    # if kfolds is applied
    elif split_method == 'kfolds':
        X_tr, X_te = np.array(datainput['x-train']), np.array(datainput['x-test'])
        Y_tr, Y_te = np.array(datainput['y-train']), np.array(datainput['y-test'])
        recall_score_list = []
        n_iter = 0
        for train_index, test_index in datainput['kfold'].split(X_tr):
            model.fit(X_tr[train_index], Y_tr[train_index])
            # accuracy measures
            test_y = model.predict(X_tr[test_index])
            recall_accuracy = recall_score(Y_tr[test_index], test_y)
            recall_score_list.append(recall_accuracy) 
            print('Iteration (%s),  recall score = %s' % (n_iter, recall_accuracy))
            n_iter += 1
            
            # logging
        print 
        print 'Mean recall score ', np.mean(recall_score_list)
        print "\n"
        
        accaracy_measures(model, datainput, conf_mat=True, roc_curve=True)


if __name__ == "__main__":
    train_on_data(
        *load_dataset(dataset_split_method='kfolds', sampling_method=None, train_test_ratio=0.3),
        model=RandomForestClassifier(n_estimators=2, n_jobs=-1)
    )