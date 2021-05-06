import os
import time

import numpy as np
import yaml
from scipy.stats import kurtosis
from scipy.stats import skew
from sklearn.metrics import accuracy_score, confusion_matrix
from sktime.classification.all import (
    ElasticEnsemble, KNeighborsTimeSeriesClassifier, ProximityForest, ROCKETClassifier)
from sktime.classification.hybrid import HIVECOTEV1
from sktime.utils.data_processing import from_2d_array_to_nested

from data import HapticDataset


def get_central_moments(x, axis=1):
    mean = np.mean(x, axis=axis, keepdims=True)
    var = np.var(x, axis=axis, keepdims=True)
    skewness = skew(x, axis=axis)[:, np.newaxis, :]
    tailedness = kurtosis(x, axis=axis)[:, np.newaxis, :]
    metrics = np.concatenate([mean, var, skewness, tailedness], axis=axis)
    return np.reshape(metrics, [metrics.shape[0], metrics.shape[1] * metrics.shape[2]])


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
    x_train = np.asarray([(s['signal'][train_ds.signal_start: train_ds.signal_start + train_ds.signal_length]
                           - train_ds.mean) / train_ds.std for s in train_ds.signals])
    x_test = np.asarray([(s['signal'][val_ds.signal_start: val_ds.signal_start + val_ds.signal_length]
                          - val_ds.mean) / val_ds.std for s in val_ds.signals])

    x_train = get_central_moments(x_train)
    x_test = get_central_moments(x_test)
    x_train = from_2d_array_to_nested(x_train)
    x_test = from_2d_array_to_nested(x_test)
    y_train = np.asarray([s['label'] for s in train_ds.signals])
    y_test = np.asarray([s['label'] for s in val_ds.signals])

    # setup the classification pipeline for tested classifiers
    classifiers = (
        KNeighborsTimeSeriesClassifier(n_neighbors=3, n_jobs=10),
        ROCKETClassifier(),
        ProximityForest(n_jobs=10),
        ElasticEnsemble(n_jobs=10),
        HIVECOTEV1(n_jobs=10)
    )

    # run train / test
    for i, clf in enumerate(classifiers):
        print("********************************")
        print("Running:\n{}\n".format(clf))

        # fit the classifier
        tic = time.time()
        clf.fit(x_train, y_train)
        toc = time.time()
        print("Elapsed fit time: {}".format(toc - tic))

        # test
        tic = time.time()
        y_pred = clf.predict(x_test)
        toc = time.time()
        print("Elapsed predict time: {}".format(toc - tic))

        # log results
        print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))
        print("Confusion matrix:\n{}".format(confusion_matrix(y_test, y_pred)))
        print("********************************")


if __name__ == '__main__':
    main()
