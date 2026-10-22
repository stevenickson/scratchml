import numpy as np


def reduce_to(x, target_shape):
    if np.broadcast_shapes(target_shape, x.shape) != x.shape:
        raise ValueError("Incorrect reduction shape")

    padded_shape = (1,) * (len(x.shape) - len(target_shape)) + target_shape
    reducible_axes = tuple(
        index for index, dimension in enumerate(padded_shape) if dimension == 1
    )
    result = x.sum(reducible_axes)
    result = np.reshape(result, target_shape)
    return result
