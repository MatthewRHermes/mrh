#!/usr/bin/env python

import inspect

import numpy as np
from scipy.sparse import linalg as sparse_linalg

# Author: Bhavnesh Jangid


"""Wrappers for SciPy's real iterative solvers with complex vector storage.

The k-LASSCF Hessian is real-linear: real and imaginary components are
independent real coordinates, so SciPy's complex-linear Krylov solvers cannot
be applied directly. These wrappers use a real problem and convert solutions
back to complex storage. By default it has 2n coordinates; an imaginary_mask
can omit inactive imaginary partners without changing the storage layout.
"""


def _tolerance_kwargs(solver, rtol, atol=None):
    """Translate tolerances across the SciPy tol/rtol API change."""
    parameters = inspect.signature(solver).parameters
    relative_key = "rtol" if "rtol" in parameters else "tol"
    kwargs = {relative_key: rtol}
    if atol is not None and "atol" in parameters:
        kwargs["atol"] = atol
    return kwargs


class SolveScipyCGForCplx:
    """Solve H x = -g for a real-linear Hessian and complex vectors.

    Parameters
    ----------
    hessian
        Callable or operator providing matvec/_matvec.
    real_hdiag : array_like, optional
        Doubled-real diagonal ordered as real coordinates followed by
        imaginary coordinates, or its compact restriction to enabled slots.
    rtol, atol, maxiter
        Convergence settings passed to :func:`scipy.sparse.linalg.cg`.
    callback : callable, optional
        Called after each iteration with the current complex step.
    compute_residual : bool, optional
        Compute ||H x + g|| after convergence.  This costs one additional
        Hessian action.
    diagonal_floor : float, optional
        Minimum absolute diagonal used by the preconditioner.
    imaginary_mask : array_like of bool, optional
        One flag per complex slot. False omits its imaginary component from
        the real solver while retaining a uniform complex storage buffer.
    """

    def __init__(
            self, hessian, real_hdiag=None, *, rtol=1e-5, atol=0.0,
            maxiter=None, callback=None, compute_residual=False,
            diagonal_floor=1e-8, imaginary_mask=None):
        
        self.imaginary_mask = imaginary_mask
        self._real_mask = None
        self.hessian = hessian
        self.real_hdiag = real_hdiag
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.maxiter = maxiter
        self.callback = callback
        self.compute_residual = bool(compute_residual)
        self.diagonal_floor = float(diagonal_floor)
        self.real_operator = None
        self.real_preconditioner = None
        self.info = None
        self.solution = None
        self.residual_norm = None

    def __call__(self, gradient, x0=None):
        """Solve the equation and return (complex_step, scipy_info)."""
        return self.run(gradient, x0=x0)

    @staticmethod
    def unpack_complex(vector):
        """Convert n complex entries into 2n real entries."""
        vector = np.asarray(vector).reshape(-1)
        if not np.all(np.isfinite(vector)):
            raise ValueError("complex vector must contain only finite values")
        return np.concatenate((vector.real, vector.imag))

    @staticmethod
    def pack_real(vector):
        """Convert [real parts, imaginary parts] to complex storage."""
        vector = np.asarray(vector)
        if vector.ndim != 1:
            raise ValueError("real-coordinate vector must be one-dimensional")
        if vector.size % 2:
            raise ValueError(
                "real-coordinate vector must have an even number of entries"
            )
        if np.iscomplexobj(vector):
            raise TypeError("real-coordinate vector must have a real dtype")
        if not np.all(np.isfinite(vector)):
            raise ValueError(
                "real-coordinate vector must contain only finite values"
            )
        ncomplex = vector.size // 2
        return (
            np.asarray(vector[:ncomplex], dtype=float)
            + 1.0j * np.asarray(vector[ncomplex:], dtype=float)
        )

    def _set_coordinate_mask(self, ncomplex):
        if self.imaginary_mask is None:
            imaginary = np.ones(ncomplex, dtype=bool)
        else:
            imaginary = np.asarray(self.imaginary_mask, dtype=bool)
            if imaginary.shape != (ncomplex,):
                raise ValueError("imaginary_mask must have one entry per complex slot")
        self._real_mask = np.concatenate((np.ones(ncomplex, dtype=bool), imaginary))

    def _unpack_coordinates(self, vector):
        doubled = self.unpack_complex(vector)
        if np.any(np.abs(doubled[~self._real_mask]) > 1e-12):
            raise ValueError("inactive imaginary coordinates must be zero")
        return doubled[self._real_mask]

    def _pack_coordinates(self, vector):
        vector = np.asarray(vector)
        if vector.shape != (int(self._real_mask.sum()),):
            raise ValueError("compact real vector has an incompatible shape")
        if np.iscomplexobj(vector) or not np.all(np.isfinite(vector)):
            raise ValueError("compact real vector must contain finite real values")
        doubled = np.zeros(self._real_mask.size, dtype=float)
        doubled[self._real_mask] = vector
        return self.pack_real(doubled)

    def _complex_matvec(self, vector):
        matvec = getattr(self.hessian, "matvec", None)
        if matvec is None:
            matvec = getattr(self.hessian, "_matvec", None)
        if matvec is None:
            if not callable(self.hessian):
                raise TypeError(
                    "hessian must be callable or provide matvec/_matvec"
                )
            matvec = self.hessian
        result = np.asarray(matvec(vector)).reshape(-1)
        if result.size != vector.size:
            raise ValueError(
                f"Hessian action returned {result.size} entries; expected "
                f"{vector.size}"
            )
        if not np.all(np.isfinite(result)):
            raise ValueError("Hessian action returned non-finite values")
        return result

    def _make_real_operator(self, ncomplex):
        def matvec(real_vector):
            complex_vector = self._pack_coordinates(real_vector)
            complex_result = self._complex_matvec(complex_vector)
            return self._unpack_coordinates(complex_result)

        return sparse_linalg.LinearOperator(
            (int(self._real_mask.sum()),) * 2, matvec=matvec, dtype=float,
        )

    def _make_real_preconditioner(self, ncomplex):
        if self.real_hdiag is None:
            return None

        diagonal = np.asarray(self.real_hdiag)
        nreal = int(self._real_mask.sum())
        if diagonal.ndim != 1 or diagonal.size not in (2 * ncomplex, nreal):
            raise ValueError(
                "real_hdiag must be a one-dimensional doubled-real or compact "
                f"diagonal of size {2 * ncomplex} or {nreal}; got {diagonal.shape}"
            )
        if np.iscomplexobj(diagonal):
            raise TypeError("real_hdiag must have a real dtype")
        diagonal = np.asarray(diagonal, dtype=float).copy()
        if diagonal.size == 2 * ncomplex:
            diagonal = diagonal[self._real_mask]
        if not np.all(np.isfinite(diagonal)):
            raise ValueError("real_hdiag must contain only finite values")

        small = np.abs(diagonal) < self.diagonal_floor
        signs = np.where(diagonal[small] < 0.0, -1.0, 1.0)
        diagonal[small] = signs * self.diagonal_floor

        return sparse_linalg.LinearOperator(
            (nreal, nreal),
            matvec=lambda vector: vector / diagonal,
            dtype=float,
        )

    def _prepare_solve(self, gradient, x0):
        gradient = np.asarray(gradient).reshape(-1)
        if gradient.size == 0:
            raise ValueError("gradient must contain at least one entry")
        if not np.all(np.isfinite(gradient)):
            raise ValueError("gradient must contain only finite values")

        ncomplex = gradient.size
        self._set_coordinate_mask(ncomplex)
        self.real_operator = self._make_real_operator(ncomplex)
        self.real_preconditioner = self._make_real_preconditioner(ncomplex)
        rhs = -self._unpack_coordinates(gradient)

        if x0 is None:
            real_x0 = None
        else:
            x0 = np.asarray(x0).reshape(-1)
            if x0.size != ncomplex:
                raise ValueError(
                    f"x0 has {x0.size} entries; expected {ncomplex}"
                )
            real_x0 = self._unpack_coordinates(x0)

        if self.callback is None:
            real_callback = None
        else:
            real_callback = lambda vector: self.callback(
                self._pack_coordinates(vector)
            )

        return gradient, rhs, real_x0, real_callback

    def _finish_solve(self, real_solution, gradient):
        self.solution = self._pack_coordinates(real_solution)
        self.residual_norm = None
        if self.compute_residual:
            residual = self._complex_matvec(self.solution) + gradient
            self.residual_norm = float(np.linalg.norm(residual))
        return np.array(self.solution, copy=True), self.info

    def run(self, gradient, x0=None):
        """Solve the doubled-real problem with conjugate gradients."""
        gradient, rhs, real_x0, real_callback = self._prepare_solve(
            gradient, x0,
        )

        solver_kwargs = _tolerance_kwargs(
            sparse_linalg.cg, self.rtol, self.atol,
        )
        real_solution, self.info = sparse_linalg.cg(
            self.real_operator,
            rhs,
            x0=real_x0,
            maxiter=self.maxiter,
            M=self.real_preconditioner,
            callback=real_callback,
            **solver_kwargs,
        )
        return self._finish_solve(real_solution, gradient)


class SolveScipyMINRESForCplx(SolveScipyCGForCplx):
    """Solve a symmetric, possibly indefinite doubled-real problem."""

    def run(self, gradient, x0=None):
        """Solve the doubled-real problem with MINRES."""
        gradient, rhs, real_x0, real_callback = self._prepare_solve(
            gradient, x0,
        )

        solver_kwargs = _tolerance_kwargs(
            sparse_linalg.minres, self.rtol,
        )
        real_solution, self.info = sparse_linalg.minres(
            self.real_operator,
            rhs,
            x0=real_x0,
            maxiter=self.maxiter,
            M=self.real_preconditioner,
            callback=real_callback,
            **solver_kwargs,
        )
        return self._finish_solve(real_solution, gradient)
