

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


pd.set_option('display.width', 210)

def load_dataset(filename="creditcard.csv", split_ratio=):
    """
    interface function to load the dataset and perform cross-validation splitting
    """
    dataset = pd.read_csv(filename)
    try:
        store = pd.HDFStore('creditcard.h5')
        store['creditcard'] = dataset
    except:
        print("[WARNING] could not save to hdfs")

    features_labels = ['V%i' % i for i in range(1, 21)]
    features = dataset[features_labels]
    classes = dataset['Class']

    # cross-validation
    X_train, X_test, y_train, y_test = train_test_split(features, classes, test_size=split_ratio, random_state=0)

    return {'x-train': X_train, 'y-train': y_train, 'x-test': X_test, 'y-test': y_test}


def model(dataset):
    """
    """
    log_reg = LogisticRegression(C=0.1, penalty='l1', tol=0.01)
    log_reg.fit(dataset['x-train'], dataset['y-train'])

    print log_reg.score(
        dataset['x-test'],
        dataset['y-test'])

ds = load_dataset()
model(ds)