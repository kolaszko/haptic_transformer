import os
import pickle

from torch.utils.data import Dataset

from submodules.haptic_transformer.data import HapticDataset, QCATDataset


def load_dataset(config):
    if config["dataset_type"].lower() == "put":
        dataset_path = os.path.join(config['dataset_folder'], config['dataset_file'])
        train_ds = HapticDataset(dataset_path, 'train_ds',
                                 signal_start=config['signal_start'],
                                 signal_length=config['signal_length'],
                                 pick_modalities=config['pick_modalities'],
                                 split_modalities=config['split_modalities'])

        val_ds = HapticDataset(dataset_path, 'val_ds',
                               signal_start=config['signal_start'],
                               signal_length=config['signal_length'],
                               pick_modalities=config['pick_modalities'],
                               split_modalities=config['split_modalities'])

        test_ds = HapticDataset(dataset_path, 'test_ds',
                                signal_start=config['signal_start'],
                                signal_length=config['signal_length'],
                                pick_modalities=config['pick_modalities'],
                                split_modalities=config['split_modalities'])

    elif config["dataset_type"].lower() == "qcat":
        if "from_data" in config.keys() and config["from_data"] == True:
            train_ds = QCATDataset.from_data(config['dataset_folder'], 'train_ds',
                                             signal_start=config['signal_start'],
                                             signal_length=config['signal_length'],
                                             pick_modalities=config['pick_modalities'],
                                             split_modalities=config['split_modalities'])
            val_ds = QCATDataset.from_data(config['dataset_folder'], 'val_ds',
                                           signal_start=config['signal_start'],
                                           signal_length=config['signal_length'],
                                           pick_modalities=config['pick_modalities'],
                                           split_modalities=config['split_modalities'])
            test_ds = QCATDataset.from_data(config['dataset_folder'], 'test_ds',
                                            signal_start=config['signal_start'],
                                            signal_length=config['signal_length'],
                                            pick_modalities=config['pick_modalities'],
                                            split_modalities=config['split_modalities'])
        else:
            dataset_path = os.path.join(config['dataset_folder'], config['dataset_file'])
            train_ds = QCATDataset.from_pickle(dataset_path, 'train_ds',
                                               signal_start=config['signal_start'],
                                               signal_length=config['signal_length'],
                                               pick_modalities=config['pick_modalities'],
                                               split_modalities=config['split_modalities'])
            val_ds = QCATDataset.from_pickle(dataset_path, 'val_ds',
                                             signal_start=config['signal_start'],
                                             signal_length=config['signal_length'],
                                             pick_modalities=config['pick_modalities'],
                                             split_modalities=config['split_modalities'])
            test_ds = QCATDataset.from_pickle(dataset_path, 'test_ds',
                                              signal_start=config['signal_start'],
                                              signal_length=config['signal_length'],
                                              pick_modalities=config['pick_modalities'],
                                              split_modalities=config['split_modalities'])

    else:
        raise NotImplementedError("Dataset not recognized. Allowed options are: QCAT, PUT")

    return train_ds, val_ds, test_ds


def load_samples_to_device(data, device):
    if type(data[0]) is list:
        s = [s.to(device).float() for s in data[0]]  # in case of split modalities return a list
    else:
        s = data[0].to(device).float()
    labels = data[-1].to(device)
    return s, labels


def concatenate_datasets(datasets: list):
    assert type(datasets) is list
    assert len(datasets) > 0
    for ds in datasets:
        assert isinstance(ds, Dataset)

    # cannot use ConcatDataset from torch, because then we loose functionality from our dataet wrappers
    # thats why we need custom concatenate method
    full_ds = datasets[0]
    for ds in datasets[1:]:
        full_ds += ds  # __add__ method implemented
    return full_ds


def save_as_pickle(path: str, dataset: dict):
    with open(path, 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
