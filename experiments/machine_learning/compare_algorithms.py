import os
import time

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix
from sktime.classification.all import (KNeighborsTimeSeriesClassifier, ProximityForest, ROCKETClassifier, BOSSEnsemble)
from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.utils.data_processing import from_3d_numpy_to_nested
from sktime.distances.elastic import dtw_distance

from data import HapticDataset


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

    x_train = from_3d_numpy_to_nested(x_train.transpose((0, 2, 1)))
    x_test = from_3d_numpy_to_nested(x_test.transpose((0, 2, 1)))
    y_train = np.asarray([s['label'] for s in train_ds.signals])
    y_test = np.asarray([s['label'] for s in val_ds.signals])

    classifiers = (
        ProximityForest(n_jobs=10, distance_measure=dtw_distance),
        BOSSEnsemble(n_jobs=10),
        ROCKETClassifier(),
        KNeighborsTimeSeriesClassifier(n_neighbors=3, n_jobs=10)
    )

    # run train / test
    for i, c in enumerate(classifiers):
        print("********************************")
        print("Running:\n{}\n".format(c))

        # apply classifier for each column or apply classifier for the multivariate data
        if c.capabilities["multivariate"]:
            clf = c
        else:
            estimators = [("{}".format(k), c, [k]) for k in range(x_train.shape[-1])]
            clf = ColumnEnsembleClassifier(
                estimators=estimators
            )

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
