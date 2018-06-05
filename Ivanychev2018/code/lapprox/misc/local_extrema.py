import numpy as np

def find_local_extrema(array: np.ndarray, radius: int):
    output = np.array([False] * len(array))
    for idx, val in enumerate(array):
        indexes = np.array([x
                            for x in range(idx - radius, idx + radius + 1)
                            if x > 0 and x < len(array)])
        output[idx] = array[idx] >= np.max(array[indexes])
    return output