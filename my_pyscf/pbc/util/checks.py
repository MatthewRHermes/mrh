"""Small input-validation helpers shared across the pbc modules."""

import numpy as np


def _check_shape(mat, shape, label="array"):
    """Validate the shape of an array-like object.

    Args:
        mat : array-like
            Object whose shape is checked.
        shape : tuple of int
            Required shape.

    Kwargs:
        label : str, optional
            Name used to identify mat in the error message.

    Raises:
        ValueError
            If the shape of mat differs from shape.
    """
    shape = tuple(shape)
    if np.shape(mat) != shape:
        msg = f"{label} has shape {np.shape(mat)}; expected {shape}"
        raise ValueError(msg)
