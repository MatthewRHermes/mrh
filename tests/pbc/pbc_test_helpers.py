"""Shared helpers for the pbc k-point transformation tests."""

import numpy as np


def phase_and_momenta(mesh, ncas, sign=1):
    """Standard Fourier mo_phase (nkpts, ncas, nkpts*ncas) and kconserv map.

    Written independently of mrh.my_pyscf.pbc.util.casdm2_transformation so
    the tests do not validate the library against itself.
    """
    cells = np.array(list(np.ndindex(mesh)))
    nkpts = len(cells)
    phase = np.zeros((nkpts, ncas, nkpts * ncas), dtype=complex)
    fourier = np.exp(sign * 2j * np.pi * (cells / np.array(mesh)) @ cells.T) / np.sqrt(nkpts)
    for orbital in range(ncas):
        phase[:, orbital, orbital::ncas] = fourier
    labels = (cells[:, None, None] - cells[None, :, None] + cells[None, None, :]) % np.array(mesh)
    kconserv = np.ravel_multi_index(labels.transpose(3, 0, 1, 2), mesh)
    return phase, kconserv
