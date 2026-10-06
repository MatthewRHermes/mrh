"""Shared k-space layout validation used by periodic kLAS-PDFT."""

import numpy as np


def _validate_kspace_layout(nkpts, ncas, kconserv=None):
    """Validate dimensions shared by the k-space RDM converters."""
    nkpts = int(nkpts)
    ncas = int(ncas)
    if nkpts <= 0:
        raise ValueError("nkpts must be positive")
    if ncas <= 0:
        raise ValueError("ncas must be positive")

    if kconserv is not None:
        kconserv = np.asarray(kconserv)
        expected_shape = (nkpts, nkpts, nkpts)
        if kconserv.shape != expected_shape:
            raise ValueError(
                f"Expected kconserv shape {expected_shape}, "
                f"got {kconserv.shape}",
            )
        if not np.issubdtype(kconserv.dtype, np.integer):
            raise ValueError("kconserv must contain integer indices")
        if np.any(kconserv < 0) or np.any(kconserv >= nkpts):
            raise ValueError("kconserv indices must lie in [0, nkpts)")
    return nkpts, ncas, kconserv
