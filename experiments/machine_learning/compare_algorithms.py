import os
import time

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sktime.classification.all import (
    BOSSEnsemble, ElasticEnsemble, KNeighborsTimeSeriesClassifier, ProximityForest, ROCKETClassifier)
from sktime.classification.hybrid import HIVECOTEV1
from sktime.transformations.panel.compose import ColumnConcatenator

from data import HapticDataset


def pandify_array(data):
    n_dim = 1
    if type(data[0]) is np.ndarray and len(data[0].shape) > 1:
        n_dim = data[0].shape[-1]

    # each modality as a separate pd.Series
    columns = ["dim_{}".format(i) for i in range(n_dim)]
    x = pd.DataFrame(columns=columns)
    for i, signal in enumerate(data):
        row = list()
        for d in range(len(columns)):
            signal_column = pd.Series(signal[:, d])
            row.append(signal_column)
        x.loc[i] = row
    return x


def main():
    with open("./config.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # load dataset
    dataset_path = os.path.join(config['dataset_folder'], config['dataset_file'])
    train_ds = HapticDataset(dataset_path, 'train_ds',
                             signal_start=config['signal_start'],
                             signal_length=config['signal_length'])

    val_ds = HapticDataset(dataset_path, 'val_ds',
                           signal_start=config['signal_start'],
                           signal_length=config['signal_length'])

    # prepare signalsL standardize and create pd.Series
    x_train = pandify_array([(s['signal'][train_ds.signal_start: train_ds.signal_start + train_ds.signal_length]
                              - train_ds.mean) / train_ds.std for s in train_ds.signals])
    x_test = pandify_array([(s['signal'][val_ds.signal_start: val_ds.signal_start + val_ds.signal_length]
                             - val_ds.mean) / val_ds.std for s in val_ds.signals])
    y_train = pd.Series([s['label'] for s in train_ds.signals])
    y_test = pd.Series([s['label'] for s in val_ds.signals])

    # setup the classification pipeline for tested classifiers
    classifier_configs = (
        tuple((('concatenate', ColumnConcatenator()), ('classify', KNeighborsTimeSeriesClassifier(n_neighbors=3, n_jobs=10)))),
        tuple((('concatenate', ColumnConcatenator()), ('classify', ProximityForest(n_jobs=10)))),
        tuple((('concatenate', ColumnConcatenator()), ('classify', ElasticEnsemble(n_jobs=10)))),
        tuple((('concatenate', ColumnConcatenator()), ('classify', HIVECOTEV1(n_jobs=10)))),
        tuple((('concatenate', ColumnConcatenator()), ('classify', BOSSEnsemble(n_jobs=10)))),
        tuple(('classify', ROCKETClassifier()))
    )

    # run train / test
    log_file = './log_{}'.format(time.time())
    for i, config in enumerate(classifier_configs):
        print("********************************", file=open(log_file, 'a'))
        print("Running:\n{}\n".format(config), file=open(log_file, 'a'))
        clf = Pipeline(config)

        # fit the classifier
        tic = time.time()
        clf.fit(x_train, y_train)
        toc = time.time()
        print("Elapsed fit time: {}".format(toc - tic), file=open(log_file, 'a'))

        # test
        tic = time.time()
        y_pred = clf.predict(x_test)
        toc = time.time()
        print("Elapsed predict time: {}".format(toc - tic), file=open(log_file, 'a'))

        # log results
        print("Accuracy: {}".format(accuracy_score(y_test, y_pred)), file=open(log_file, 'a'))
        print("Confusion matrix:\n{}".format(confusion_matrix(y_test, y_pred)), file=open(log_file, 'a'))
        print("********************************", file=open(log_file, 'a'))
        print("{} Done!\n".format(config))


if __name__ == '__main__':
    main()
