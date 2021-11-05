import os

from data import HapticDataset, QCATDataset


def load_dataset(config):
    if config["dataset_type"].lower() == "put":
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

    elif config["dataset_type"].lower() == "qcat":
        dataset_path = os.path.join(config['dataset_folder'], config['dataset_file'])
        train_ds = QCATDataset(dataset_path, 'train_ds',
                               signal_start=config['signal_start'],
                               signal_length=config['signal_length'])

        val_ds = QCATDataset(dataset_path, 'val_ds',
                             signal_start=config['signal_start'],
                             signal_length=config['signal_length'])

        test_ds = QCATDataset(dataset_path, 'test_ds',
                              signal_start=config['signal_start'],
                              signal_length=config['signal_length'])
    else:
        raise NotImplementedError("Dataset not recognized. Allowed options are: QCAT, PUT")

    return train_ds, val_ds, test_ds