"""Real coordinates for nonredundant, k-diagonal active LAS rotations."""

import numpy as np
from pyscf import lib
from mrh.util.la import safe_svd_warner


def _pivoted_projector_basis(sub_bas):
    """Choose stable coordinates from an orthonormal subspace projector."""
    nrow, rank = sub_bas.shape
    if rank == 0:
        return sub_bas.copy()
    projector = sub_bas @ sub_bas.conj().T
    residual = projector.copy()
    basis = np.empty((nrow, rank), dtype=sub_bas.dtype)
    for column in range(rank):
        norms = np.maximum(np.diag(residual).real, 0.)
        largest = np.max(norms)
        pivot = np.flatnonzero(norms >= largest * (1 - 1e-10))[0]
        vec = projector[:, pivot].copy()
        for _ in range(2):
            retained = basis[:, :column]
            vec -= retained @ (retained.conj().T @ vec)
        vec /= np.linalg.norm(vec)
        vec *= vec[pivot].conj() / abs(vec[pivot])
        basis[:, column] = vec
        residual -= np.outer(vec, vec.conj())
    return basis


class ActiveActiveRotationMap:
    r"""Complete anti-Hermitian Bloch generators modulo LAS redundancy.

    Coordinates have REAL values (returned in a complex storage buffer), with the real Frobenius inner product
    ``Re trace(A.conj().T B)`` summed over k-points. Each selected lower pair
    supplies ``(E_ab-E_ba)/sqrt(2)`` and ``i(E_ab+E_ba)/sqrt(2)``;
    each selected diagonal supplies ``i E_aa``. These generators have unit
    Frobenius norm. ``nvar`` counts real degrees of freedom, not complex pairs.

    Transform allowed generators to Wannier space, retain their inter-fragment
    lower entries (real and imaginary parts), and compress this real-linear
    map by SVD. The retained Bloch subspace is the orthogonal complement of
    generators whose Wannier matrices are entirely intra-fragment. This
    removes precisely the redundancy within the allowed k-diagonal space.

    For Fourier-related equal fragments, retained Wannier matrices have zero
    intra-fragment blocks. With a general unitary ``mo_phase``, that need not
    hold: projecting away intra-fragment blocks can leave the allowed Bloch
    space. We remove the intersection with the redundant space instead.

    ``mo_phase`` has shape ``(nkpts, ncas, nkpts*ncas)`` and its stacked rows
    must be unitary. ``ncas_sub`` gives the Wannier fragment sizes. The optional
    ``bloch_pair_mask`` has shape ``(nkpts,ncas,ncas)`` and selects LOWER pairs
    AND diagonal entries; an all-true lower triangle is the default. A frozen
    band is excluded by clearing its row and column. No time-reversal condition
    is imposed: arbitrary complex gauges are supported.

    ``basis`` has orthonormal real columns in the complete selected Bloch
    coordinate space. ``pair_map`` maps real inter-fragment Wannier coordinates
    into that space; its image defines ``basis``. ``pack`` is both the rotation
    projection and the adjoint of ``unpack`` in the stated Frobenius metric.
    Gradient callers must account for their own energy normalization.
    """

    def __init__(self, mo_phase, ncas_sub, bloch_pair_mask=None,
                 svd_tol=None, verbose=None):
        phase = np.asarray(mo_phase, dtype=complex)
        if phase.ndim != 3:
            raise ValueError('mo_phase must have shape (nkpts, ncas, ncastot)')
        self.nkpts, self.ncas, self.ncastot = phase.shape
        if self.nkpts < 1 or self.ncastot != self.nkpts * self.ncas:
            raise ValueError('mo_phase must map a square stacked Bloch-active space')
        stacked = phase.reshape(self.ncastot, self.ncastot)
        if not np.isfinite(phase).all() or not np.allclose(
                stacked.conj().T @ stacked, np.eye(self.ncastot),
                rtol=0., atol=1e-10):
            raise ValueError('mo_phase must be unitary')
        sizes = np.asarray(ncas_sub)
        if (sizes.ndim != 1 or not np.issubdtype(sizes.dtype, np.integer)
                or np.any(sizes <= 0) or sizes.sum() != self.ncastot):
            raise ValueError('ncas_sub must contain positive integer fragment sizes '
                             'summing to ncastot')
        if svd_tol is not None and (not np.isfinite(svd_tol) or svd_tol < 0):
            raise ValueError('svd_tol must be finite and nonnegative')
        self.mo_phase = phase.copy()
        self.ncas_sub = sizes.astype(int, copy=True)
        shape = (self.nkpts, self.ncas, self.ncas)
        if bloch_pair_mask is None:
            bloch_pair_mask = np.broadcast_to(
                np.tril(np.ones((self.ncas, self.ncas), dtype=bool)), shape)
        mask = np.asarray(bloch_pair_mask, dtype=bool)
        if mask.shape != shape:
            raise ValueError(f'bloch_pair_mask has shape {mask.shape}; expected {shape}')
        if np.any(np.triu(mask, 1)):
            raise ValueError('bloch_pair_mask may select only lower pairs and diagonals')
        self.bloch_pair_mask = mask.copy()
        self.bloch_pair_idx = np.where(mask)
        # Explicit metadata fixes coordinate order and avoids fictitious
        # imaginary partners for the real diagonal coordinates.
        self.generators = tuple(
            (k, a, b, kind) for k, a, b in zip(*self.bloch_pair_idx)
            for kind in (('imag',) if a == b else ('real', 'imag')))
        fragment = np.repeat(np.arange(sizes.size), sizes)
        self.wannier_pair_idx = np.where(fragment[:, None] > fragment[None, :])
        p, q = self.wannier_pair_idx
        nwannier = len(p)
        self.pair_map = np.empty((len(self.generators), 2 * nwannier), dtype=float)
        for index, (k, a, b, kind) in enumerate(self.generators):
            ab = phase[k, a, p].conj() * phase[k, b, q]
            if a == b:
                values = 1j * ab
            else:
                ba = phase[k, b, p].conj() * phase[k, a, q]
                values = ((ab - ba) if kind == 'real' else 1j * (ab + ba)) / np.sqrt(2)
            self.pair_map[index, :nwannier] = np.sqrt(2) * values.real
            self.pair_map[index, nwannier:] = np.sqrt(2) * values.imag
        log = lib.logger.new_logger(None, lib.logger.QUIET if verbose is None else verbose)
        if min(self.pair_map.shape) == 0:
            self.singular_values = np.empty(0)
            self.svd_tol = 0. if svd_tol is None else float(svd_tol)
            self.basis = np.empty((len(self.generators), 0), dtype=float)
        else:
            left, values, _ = safe_svd_warner(log.warn)(self.pair_map, full_matrices=False)
            self.singular_values = values
            self.svd_tol = (max(1e-10, max(self.pair_map.shape)
                               * np.finfo(float).eps * values[0])
                            if svd_tol is None else float(svd_tol))
            self.basis = _pivoted_projector_basis(left[:, values > self.svd_tol])
        log.debug('Real active-active rotation map: retained %d of %d Bloch directions',
                  self.nvar, len(self.generators))

    @property
    def nvar(self):
        """Number of independent REAL active-active coordinates."""
        return self.basis.shape[1]

    def _check_bloch(self, matrix):
        matrix = np.asarray(matrix)
        expected = (self.nkpts, self.ncas, self.ncas)
        if matrix.shape != expected:
            raise ValueError(f'kappa_active has shape {matrix.shape}; expected {expected}')
        return matrix

    def bloch_to_wannier(self, kappa_active):
        """Unitary transformation of a k-diagonal matrix to Wannier space."""
        matrix = self._check_bloch(kappa_active)
        return np.einsum('kap,kab,kbq->pq', self.mo_phase.conj(), matrix,
                         self.mo_phase, optimize=True)

    def wannier_to_bloch(self, kappa_wannier):
        """Transform to Bloch space, retaining only the k-diagonal blocks."""
        matrix = np.asarray(kappa_wannier)
        if matrix.shape != (self.ncastot, self.ncastot):
            raise ValueError('kappa_wannier must have shape (ncastot, ncastot)')
        return np.einsum('kap,pq,kbq->kab', self.mo_phase, matrix,
                         self.mo_phase.conj(), optimize=True)

    def pack(self, kappa_active):
        """Real Frobenius projection; also the adjoint of ``unpack``."""
        matrix = self._check_bloch(kappa_active)
        full = np.empty(len(self.generators))
        for index, (k, a, b, kind) in enumerate(self.generators):
            if a == b:
                full[index] = matrix[k, a, a].imag
            elif kind == 'real':
                full[index] = (matrix[k, a, b].real - matrix[k, b, a].real) / np.sqrt(2)
            else:
                full[index] = (matrix[k, a, b].imag + matrix[k, b, a].imag) / np.sqrt(2)
        return np.asarray(self.basis.T @ full, dtype=complex)

    def unpack(self, coordinates):
        """Expand REAL coordinates into normalized anti-Hermitian generators."""
        coordinates = np.asarray(coordinates).reshape(-1)
        if coordinates.size != self.nvar:
            raise ValueError(f'coordinates have size {coordinates.size}; expected {self.nvar}')
        if not np.isfinite(coordinates).all() or np.any(np.abs(coordinates.imag) > 1e-12):
            raise ValueError('active-active coordinates must be finite and real')
        full = self.basis @ coordinates.real
        matrix = np.zeros((self.nkpts, self.ncas, self.ncas), dtype=complex)
        for value, (k, a, b, kind) in zip(full, self.generators):
            if a == b:
                matrix[k, a, a] += 1j * value
            elif kind == 'real':
                matrix[k, a, b] += value / np.sqrt(2)
                matrix[k, b, a] -= value / np.sqrt(2)
            else:
                matrix[k, a, b] += 1j * value / np.sqrt(2)
                matrix[k, b, a] += 1j * value / np.sqrt(2)
        return matrix
