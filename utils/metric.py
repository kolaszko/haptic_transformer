import time

import numpy as np
from sktime.datatypes._panel._convert import from_3d_numpy_to_nested


def measure_inference_time_sktime(model, shape=(160, 6), repetitions=300):
    mock_x = np.zeros(shape=[1, shape[0], shape[1]])
    mock_x = from_3d_numpy_to_nested(mock_x.transpose((0, 2, 1)))

    performance = list()
    for x in range(repetitions):
        tic = time.time()
        model.predict(mock_x)
        toc = time.time()
        performance.append(toc - tic)

    return np.mean(performance), np.std(performance)
