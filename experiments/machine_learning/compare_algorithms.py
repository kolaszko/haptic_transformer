import os
import time

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from sktime.classification.all import (KNeighborsTimeSeriesClassifier, ROCKETClassifier)
from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.utils.data_processing import from_3d_numpy_to_nested

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

    # standardize datasets
    x = np.asarray([(s['signal'][train_ds.signal_start: train_ds.signal_start + train_ds.signal_length]
                     - train_ds.mean) / train_ds.std for s in train_ds.signals])
    y = np.asarray([s['label'] for s in train_ds.signals])

    x_test = np.asarray([(s['signal'][val_ds.signal_start: val_ds.signal_start + val_ds.signal_length]
                          - val_ds.mean) / val_ds.std for s in val_ds.signals])
    x_test = from_3d_numpy_to_nested(x_test.transpose((0, 2, 1)))
    y_test = np.asarray([s['label'] for s in val_ds.signals])

    # run K-fold cross validation
    kfold = KFold(n_splits=config['k_folds'], shuffle=True)

    # chosen classifiers
    classifiers = (
        ROCKETClassifier(num_kernels=10000),  # conv-based
        KNeighborsTimeSeriesClassifier(n_neighbors=3, n_jobs=10)  # non-conv
    )

    for classifier in classifiers:
        val_acc_folds, test_acc_folds = list(), list()
        for fold, (train_ids, val_ids) in enumerate(kfold.split(x, y)):

            # get dataset fold and convert it to pandas Series
            x_train, x_val = x[train_ids.astype(int)], x[val_ids.astype(int)]
            y_train, y_val = y[train_ids.astype(int)], y[val_ids.astype(int)]
            x_train = from_3d_numpy_to_nested(x_train.transpose((0, 2, 1)))
            x_val = from_3d_numpy_to_nested(x_val.transpose((0, 2, 1)))

            # apply classifier for each column or apply classifier for the multivariate data
            if classifier.capabilities["multivariate"]:
                clf = classifier
            else:
                estimators = [("{}".format(k), classifier, [k]) for k in range(x_train.shape[-1])]
                clf = ColumnEnsembleClassifier(
                    estimators=estimators
                )

            print("********************************")
            print("Running:\n{}\n".format(classifier))

            # train
            tic = time.time()
            clf.fit(x_train, y_train)
            toc = time.time()
            print("Elapsed fit time: {}".format(toc - tic))

            # test
            tic = time.time()
            y_val_pred = clf.predict(x_val)
            toc = time.time()
            print("Elapsed predict time: {}".format(toc - tic))

            # log results
            y_test_pred = clf.predict(x_test)
            val_acc = accuracy_score(y_val, y_val_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            print("Val Accuracy: {}".format(val_acc))
            print("Test Accuracy: {}".format(test_acc))
            print("Confusion matrix:\n{}\n".format(confusion_matrix(y_val, y_val_pred)))
            val_acc_folds.append(val_acc)
            test_acc_folds.append(test_acc)

        print("{}-fold cross-validation summary for {}:\n".format(config['k_folds'], classifier))
        print("Mean val accuracy: {}".format(sum(val_acc_folds) / len(val_acc_folds)))
        print("Max val accuracy: {}".format(max(val_acc_folds)))
        print("Val standard deviation: {}".format(np.std(val_acc_folds)))
        print("Mean test accuracy: {}".format(sum(test_acc_folds) / len(test_acc_folds)))
        print("Max test accuracy: {}".format(max(test_acc_folds)))
        print("Test standard deviation: {}".format(np.std(test_acc_folds)))
        print("********************************")


if __name__ == '__main__':
    main()
