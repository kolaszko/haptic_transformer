import time

import numpy as np
from sktime.datatypes._panel._convert import from_3d_numpy_to_nested


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


def to_nested_3d(ds, ds_type):
    if "qcat" in ds_type.lower():
        return _qcat_to_nested(ds), _qcat_labels(ds)
    elif "put" in ds_type.lower():
        return _put_to_nested(ds), _put_labels(ds)
    else:
        raise NotImplementedError("Dataset not recognized.")


def _qcat_to_nested(ds):
    return from_3d_numpy_to_nested(ds.signals['ft'].transpose((0, 2, 1)))


def _qcat_labels(ds):
    return np.asarray(ds.signals['label'])


def _put_to_nested(ds):
    x_train = np.asarray([s['signal'] for s in ds.signals])
    return from_3d_numpy_to_nested(x_train.transpose((0, 2, 1)))


def _put_labels(ds):
    return np.asarray([s['label'] for s in ds.signals])
