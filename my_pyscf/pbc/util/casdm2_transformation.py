"""Batched Wannier-to-Bloch transformation of the 2-RDM or cumulant.

For a k-point triple (k1, k2, k3) with k4 = kconserv[k1, k2, k3]:

    einsum('iR,jS,kT,lU,RSTU->ijkl',
           P[k1].conj(), P[k2], P[k3].conj(), P[k4], casdm2)

P = mo_phase has shape (nkpts, ncas, nkpts*ncas). i,j,k,l are active orbitals
inside one Bloch block; R,S,T,U run over all Wannier orbitals. Instead of one
transform per block (nkpts^3 of them), each axis is transformed once for all k
and the conserving blocks are gathered afterwards.

For the standard Fourier phases on a regular k-mesh (checked at run time), the
tensor is first summed over a common cell shift and then transformed with
FFTs, which scale as N log N instead of N^2. Any other phases use the dense
path. Both give the same result.

Output layout: (k1, k2, k3, i, j, k, l).
"""

import numpy as np
from scipy import fft


def _fourier_phase(kmesh, ncas, sign=1):
    coords = np.array(list(np.ndindex(tuple(kmesh))))
    nkpts = len(coords)
    matrix = np.exp(sign * 2j * np.pi * (coords / np.asarray(kmesh)) @ coords.T)
    phase = np.zeros((nkpts, ncas, nkpts * ncas), dtype=np.complex128)
    for orbital in range(ncas):
        phase[:, orbital, orbital::ncas] = matrix / np.sqrt(nkpts)
    return phase


def _momentum_map(kmesh):
    coords = np.array(list(np.ndindex(tuple(kmesh))))
    indices = (coords[:, None, None] - coords[None, :, None] + coords[None, None, :]) % np.asarray(kmesh)
    return np.ravel_multi_index(indices.transpose(3, 0, 1, 2), tuple(kmesh))


def _gather(full, nkpts, ncas, kconserv):
    blocks = full.reshape((nkpts, ncas) * 4).transpose(0, 2, 4, 6, 1, 3, 5, 7)
    k1, k2, k3 = np.ogrid[:nkpts, :nkpts, :nkpts]
    return blocks[k1, k2, k3, kconserv]


def _transform_dense(casdm2, mo_phase, kconserv):
    """Transform each tensor axis once for all Bloch orbitals, then gather."""
    nkpts, ncas, nactive = mo_phase.shape
    matrix = mo_phase.reshape(nkpts * ncas, nactive)
    result = casdm2
    for axis, factor in enumerate((matrix.conj(), matrix, matrix.conj(), matrix)):
        result = np.moveaxis(np.tensordot(factor, result, axes=(1, axis)), 0, axis)
    return _gather(result, nkpts, ncas, kconserv)


def _fft_relative(relative, kmesh, ncas, sign):
    # relative layout is (R1,R2,R3,i,j,k,l). Put orbitals beside their cell axes.
    relative = relative.transpose(0, 3, 1, 4, 2, 5, 6)
    ndim_cell = len(kmesh)
    shaped = relative.reshape(tuple(kmesh) + (ncas,) + tuple(kmesh) + (ncas,)
                              + tuple(kmesh) + (ncas, ncas))
    first = tuple(range(ndim_cell))
    second = tuple(range(ndim_cell + 1, 2 * ndim_cell + 1))
    third = tuple(range(2 * ndim_cell + 2, 3 * ndim_cell + 2))
    negative, positive = (first + third, second) if sign == 1 else (second, first + third)
    result = fft.fftn(shaped, axes=negative, norm='ortho')
    result = fft.ifftn(result, axes=positive, norm='ortho', overwrite_x=True)
    nkpts = int(np.prod(kmesh))
    result /= np.sqrt(nkpts)
    return result.reshape(nkpts, ncas, nkpts, ncas, nkpts, ncas, ncas).transpose(0, 2, 4, 1, 3, 5, 6)


def _transform_fourier(casdm2, kmesh, ncas, sign=1):
    """Project onto conserving momenta, then transform only three cell axes.

    Shift all four Wannier cell indices by -R4 and sum over R4. The phase
    of that common shift cancels exactly when k1-k2+k3-k4=0. This reduces
    the transform input from nkpts**4*ncas**4 to nkpts**3*ncas**4.
    """
    kmesh = tuple(kmesh)
    coords = np.array(list(np.ndindex(kmesh)))
    nkpts = len(coords)
    blocks = casdm2.reshape((nkpts, ncas) * 4).transpose(0, 2, 4, 6, 1, 3, 5, 7)
    relative = np.zeros((nkpts,) * 3 + (ncas,) * 4, dtype=np.result_type(casdm2.dtype, np.float64))
    for r4, origin in enumerate(coords):
        shifted = np.ravel_multi_index(((coords + origin) % np.asarray(kmesh)).T, kmesh)
        relative += blocks[shifted[:, None, None], shifted[None, :, None], shifted[None, None, :], r4]
    return _fft_relative(relative, kmesh, ncas, sign)


def transform_casdm2_kpts(casdm2, mo_phase, kconserv, *, kmesh=None, method='auto'):
    """Transform a Wannier 2-RDM or cumulant to all conserving Bloch blocks.

    Args:
        casdm2: Tensor with shape (nkpts*ncas,) * 4 in cell-major order.
        mo_phase: Wannier-to-Bloch matrix, shape (nkpts, ncas, nkpts*ncas).
        kconserv: Integer map of shape (nkpts,) * 3 giving k4 for each triple.
        kmesh: Optional regular k-mesh, needed to use the Fourier path.
        method: "auto" (Fourier when valid, else dense), "dense", or
            "fourier" (raises if the Fourier path is not valid).

    Returns:
        Array of shape (nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas) whose
        [k1,k2,k3] block equals the per-block transform with
        k4 = kconserv[k1,k2,k3], up to roundoff.
    """
    casdm2 = np.asarray(casdm2)
    mo_phase = np.asarray(mo_phase)
    nkpts, ncas, nactive = mo_phase.shape
    if nactive != nkpts * ncas or casdm2.shape != (nactive,) * 4:
        raise ValueError('Expected a complete Wannier active space of nkpts*ncas orbitals')
    kconserv = np.asarray(kconserv)
    if kconserv.shape != (nkpts,) * 3 or np.any(kconserv < 0) or np.any(kconserv >= nkpts):
        raise ValueError('Invalid kconserv map')
    if method not in ('auto', 'dense', 'fourier'):
        raise ValueError('method must be auto, dense, or fourier')
    if method != 'dense' and kmesh is not None and int(np.prod(kmesh)) == nkpts:
        if np.array_equal(kconserv, _momentum_map(kmesh)):
            for sign in (1, -1):
                if np.allclose(mo_phase, _fourier_phase(kmesh, ncas, sign), atol=1e-12, rtol=1e-12):
                    return _transform_fourier(casdm2, kmesh, ncas, sign)
    if method == 'fourier':
        raise ValueError('Fourier path requires the standard mesh order and Fourier phases')
    return _transform_dense(casdm2, mo_phase, kconserv)
