import os
import time

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix
from sktime.classification.all import (KNeighborsTimeSeriesClassifier, ROCKETClassifier)
from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.utils.data_processing import from_3d_numpy_to_nested

from data import HapticDataset


def measure_inference_time(model, shape=(160, 6), repetitions=300):
    mock_x = np.zeros(shape=[1, shape[0], shape[1]])
    mock_x = from_3d_numpy_to_nested(mock_x.transpose((0, 2, 1)))

    performance = list()
    for x in range(repetitions):
        tic = time.time()
        model.predict(mock_x)
        toc = time.time()
        performance.append(toc - tic)

    return np.mean(performance), np.std(performance)


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

    test_ds = HapticDataset(dataset_path, 'test_ds',
                            signal_start=config['signal_start'],
                            signal_length=config['signal_length'])

    # feed for classifiers (pd.Series)
    x_train = np.asarray([(s['signal'] - train_ds.mean) / train_ds.std for s in train_ds.signals])
    x_train = from_3d_numpy_to_nested(x_train.transpose((0, 2, 1)))
    x_val = np.asarray([(s['signal'] - val_ds.mean) / val_ds.std for s in val_ds.signals])
    x_val = from_3d_numpy_to_nested(x_val.transpose((0, 2, 1)))
    x_test = np.asarray([(s['signal'] - test_ds.mean) / test_ds.std for s in test_ds.signals])
    x_test = from_3d_numpy_to_nested(x_test.transpose((0, 2, 1)))

    # labels
    y_train = np.asarray([s['label'] for s in train_ds.signals])
    y_val = np.asarray([s['label'] for s in val_ds.signals])
    y_test = np.asarray([s['label'] for s in test_ds.signals])

    # chosen classifiers
    classifiers = (
        ROCKETClassifier(num_kernels=10000),  # conv-based
        KNeighborsTimeSeriesClassifier(n_neighbors=3, n_jobs=10)  # non-conv
    )

    # main loop
    for classifier in classifiers:

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
        clf.fit(x_train, y_train)

        # validate & test
        y_val_pred = clf.predict(x_val)
        y_test_pred = clf.predict(x_test)

        # measure inference times
        mean_time, std_time = measure_inference_time(clf)

        # log results
        acc_val = accuracy_score(y_val, y_val_pred)
        conf_mat_val = confusion_matrix(y_val, y_val_pred)
        acc_test = accuracy_score(y_test, y_test_pred)
        conf_mat_test = confusion_matrix(y_test, y_test_pred)

        print("Validation accuracy: {}".format(acc_val))
        print("Validation confusion matrix:\n{}\n".format(conf_mat_val))
        print("Test accuracy: {}".format(acc_test))
        print("Test confusion matrix:\n{}\n".format(conf_mat_test))
        print("Inference time:\n{} +/- {}\n".format(mean_time, std_time))
        print("********************************")


if __name__ == '__main__':
    main()
