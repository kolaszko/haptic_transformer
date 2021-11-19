import argparse

import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sktime.classification.all import (KNeighborsTimeSeriesClassifier, ROCKETClassifier)

import utils


def main(args):
    with open(args.dataset_config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # load the dataset
    train_ds, val_ds, test_ds = utils.dataset.load_dataset(config)

    # feed for classifiers (pd.Series)
    x_train, y_train = utils.sktime.to_nested_3d(train_ds, config["dataset_type"])
    x_val, y_val = utils.sktime.to_nested_3d(val_ds, config["dataset_type"])
    x_test, y_test = utils.sktime.to_nested_3d(test_ds, config["dataset_type"])
    data_shape = train_ds.signal_length, train_ds.mean.shape[-1]

    # chosen classifiers
    classifiers = (
        ROCKETClassifier(num_kernels=10000),  # conv-based
        KNeighborsTimeSeriesClassifier(n_neighbors=3, n_jobs=10)  # non-conv
    )

    # main loop
    print("Run training.")
    for clf in classifiers:
        print("********************************")
        print("Running:\n{}\n".format(clf))

        # train
        clf.fit(x_train, y_train)

        # validate & test
        y_val_pred = clf.predict(x_val)
        y_test_pred = clf.predict(x_test)

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

        # measure inference times
        mean_time, std_time = utils.sktime.measure_inference_time(clf, shape=data_shape)
        print("Inference time:\n{} +/- {}\n".format(mean_time, std_time))
        print("********************************")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-config-file', type=str,
                        default="/home/mbed/Projects/haptic_transformer/experiments/config/config_put.yaml")
    args, _ = parser.parse_known_args()
    main(args)
