#!/usr/bin/env python

import inspect

import numpy as np
from scipy.sparse import linalg as sparse_linalg

# Author: Bhavnesh Jangid


"""Wrappers for SciPy's real iterative solvers with complex vector storage.

The k-LASSCF Hessian is real-linear: real and imaginary components are
independent real coordinates, so SciPy's complex-linear Krylov solvers cannot
be applied directly.  These wrappers expose an n-element complex problem as a
2n-element real problem and convert the solution back to complex storage.
"""


def _tolerance_kwargs(solver, rtol, atol=None):
    """Translate tolerances across the SciPy ``tol``/``rtol`` API change."""
    parameters = inspect.signature(solver).parameters
    relative_key = "rtol" if "rtol" in parameters else "tol"
    kwargs = {relative_key: rtol}
    if atol is not None and "atol" in parameters:
        kwargs["atol"] = atol
    return kwargs


class SolveScipyCGForCplx:
    """Solve ``H x = -g`` for a real-linear Hessian and complex vectors.

    Parameters
    ----------
    hessian
        Callable or operator providing ``matvec``/``_matvec``.
    real_hdiag : array_like, optional
        Doubled-real diagonal ordered as real coordinates followed by
        imaginary coordinates.
    rtol, atol, maxiter
        Convergence settings passed to :func:`scipy.sparse.linalg.cg`.
    callback : callable, optional
        Called after each iteration with the current complex step.
    compute_residual : bool, optional
        Compute ``||H x + g||`` after convergence.  This costs one additional
        Hessian action.
    diagonal_floor : float, optional
        Minimum absolute diagonal used by the preconditioner.
    """

    def __init__(
            self, hessian, real_hdiag=None, *, rtol=1e-5, atol=0.0,
            maxiter=None, callback=None, compute_residual=False,
            diagonal_floor=1e-8):
        self.hessian = hessian
        self.real_hdiag = real_hdiag
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.maxiter = maxiter
        self.callback = callback
        self.compute_residual = bool(compute_residual)
        self.diagonal_floor = float(diagonal_floor)

        if self.rtol < 0.0 or self.atol < 0.0:
            raise ValueError("rtol and atol must be nonnegative")
        if not np.isfinite(self.diagonal_floor) or self.diagonal_floor <= 0.0:
            raise ValueError("diagonal_floor must be finite and positive")

        self.real_operator = None
        self.real_preconditioner = None
        self.info = None
        self.solution = None
        self.residual_norm = None

    def __call__(self, gradient, x0=None):
        """Solve the equation and return ``(complex_step, scipy_info)``."""
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
        """Convert ``[real parts, imaginary parts]`` to complex storage."""
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
            complex_vector = self.pack_real(real_vector)
            complex_result = self._complex_matvec(complex_vector)
            return self.unpack_complex(complex_result)

        return sparse_linalg.LinearOperator(
            (2 * ncomplex, 2 * ncomplex), matvec=matvec, dtype=float,
        )

    def _make_real_preconditioner(self, ncomplex):
        if self.real_hdiag is None:
            return None

        diagonal = np.asarray(self.real_hdiag)
        if diagonal.ndim != 1 or diagonal.size != 2 * ncomplex:
            raise ValueError(
                "real_hdiag must be a one-dimensional doubled-real "
                f"diagonal of size {2 * ncomplex}; got {diagonal.shape}"
            )
        if np.iscomplexobj(diagonal):
            raise TypeError("real_hdiag must have a real dtype")
        diagonal = np.asarray(diagonal, dtype=float).copy()
        if not np.all(np.isfinite(diagonal)):
            raise ValueError("real_hdiag must contain only finite values")

        small = np.abs(diagonal) < self.diagonal_floor
        signs = np.where(diagonal[small] < 0.0, -1.0, 1.0)
        diagonal[small] = signs * self.diagonal_floor

        return sparse_linalg.LinearOperator(
            (2 * ncomplex, 2 * ncomplex),
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
        self.real_operator = self._make_real_operator(ncomplex)
        self.real_preconditioner = self._make_real_preconditioner(ncomplex)
        rhs = -self.unpack_complex(gradient)

        if x0 is None:
            real_x0 = None
        else:
            x0 = np.asarray(x0).reshape(-1)
            if x0.size != ncomplex:
                raise ValueError(
                    f"x0 has {x0.size} entries; expected {ncomplex}"
                )
            real_x0 = self.unpack_complex(x0)

        if self.callback is None:
            real_callback = None
        else:
            real_callback = lambda vector: self.callback(
                self.pack_real(vector)
            )

        return gradient, rhs, real_x0, real_callback

    def _finish_solve(self, real_solution, gradient):
        self.solution = self.pack_real(real_solution)
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
