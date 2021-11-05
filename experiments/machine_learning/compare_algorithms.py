import numpy as np
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sktime.classification.all import (KNeighborsTimeSeriesClassifier, ROCKETClassifier)
from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.datatypes._panel._convert import from_3d_numpy_to_nested

import utils


def main():
    with open("./config.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # load dataset
    train_ds, val_ds, test_ds = utils.dataset.load_dataset(config)

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
        mean_time, std_time = utils.metric.measure_inference_time_sktime(clf)

        # log results
        acc_val = accuracy_score(y_val, y_val_pred)
        conf_mat_val = confusion_matrix(y_val, y_val_pred, normalize='true')
        acc_test = accuracy_score(y_test, y_test_pred)
        conf_mat_test = confusion_matrix(y_test, y_test_pred, normalize='true')
        test_cls_report = classification_report(y_test, y_test_pred)

        print("Validation accuracy: {}".format(acc_val))
        print("Validation confusion matrix:\n{}\n".format(conf_mat_val))
        print("Test accuracy: {}".format(acc_test))
        print("Test confusion matrix:\n{}\n".format(conf_mat_test))
        print("Classification report:\n{}\n".format(test_cls_report))
        print("Inference time:\n{} +/- {}\n".format(mean_time, std_time))
        print("********************************")


if __name__ == '__main__':
    main()
