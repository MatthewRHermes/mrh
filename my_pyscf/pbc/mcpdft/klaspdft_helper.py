"""Reduced-density-matrix helpers for periodic kLAS-PDFT.

The kLAS wave function is a product of fragment states expressed in a
Wannier active-orbital basis.  This module selects one root from that product
state and assembles the full active-space one- and two-body density matrices
needed by MC-PDFT.  Basis transformation and energy evaluation deliberately
live outside this module.
"""

from numbers import Integral

import numpy as np

from mrh.my_pyscf.pbc.mcscf.productstate import PBCProductStateFCISolver
from mrh.my_pyscf.pbc.util.wannier import get_wannier_orbs


def _get_klas_rdm_context(klas, ci=None, state=0):
    """Resolve the fragment solvers and CI vectors for one kLAS root.

    Parameters
    ----------
    klas : object
        A completed periodic LASCI or LASSCF object.  It must provide
        ``fciboxes``, ``ncas_sub``, ``nelecas_sub``, ``nroots``, and ``ci``.
    ci : sequence, optional
        Fragment-major CI vectors.  ``ci[ifrag][state]`` is the selected CI
        vector for fragment ``ifrag``.  The stored ``klas.ci`` is used by
        default.
    state : int, optional
        Rootspace index to select.

    Returns
    -------
    fcisolvers : list
        One fragment FCI solver for the selected rootspace.
    ci_state : list
        One CI vector per fragment for the selected rootspace.
    ncas_sub : ndarray
        Numbers of active orbitals in the fragments.
    nelecas_sub : ndarray
        Alpha and beta active-electron counts in the fragments.
    """
    if not isinstance(state, Integral):
        raise TypeError("state must be an integer")
    state = int(state)

    try:
        nroots = int(klas.nroots)
    except (AttributeError, TypeError, ValueError) as err:
        raise ValueError("klas.nroots must be a positive integer") from err
    if nroots <= 0:
        raise ValueError("klas.nroots must be a positive integer")
    if state < 0 or state >= nroots:
        raise ValueError(
            f"state must lie in [0, {nroots}); got {state}",
        )

    ncas_sub = np.asarray(getattr(klas, "ncas_sub", ()), dtype=int)
    nelecas_sub = np.asarray(getattr(klas, "nelecas_sub", ()), dtype=int)
    fciboxes = list(getattr(klas, "fciboxes", ()))
    if ncas_sub.ndim != 1 or not ncas_sub.size or np.any(ncas_sub <= 0):
        raise ValueError("ncas_sub must be a nonempty vector of positive integers")
    nfrags = ncas_sub.size
    if nelecas_sub.shape != (nfrags, 2):
        raise ValueError(
            f"nelecas_sub must have shape ({nfrags}, 2); "
            f"got {nelecas_sub.shape}",
        )
    if len(fciboxes) != nfrags:
        raise ValueError(
            f"Expected {nfrags} fragment FCI boxes; got {len(fciboxes)}",
        )

    if ci is None:
        ci = getattr(klas, "ci", None)
    if ci is None:
        raise ValueError("The kLAS object has no CI vectors")
    if len(ci) != nfrags:
        raise ValueError(
            f"Expected CI vectors for {nfrags} fragments; got {len(ci)}",
        )

    fcisolvers = []
    ci_state = []
    for ifrag, (fcibox, ci_frag) in enumerate(zip(fciboxes, ci)):
        solvers = list(getattr(fcibox, "fcisolvers", ()))
        if len(solvers) <= state:
            raise ValueError(
                f"Fragment {ifrag} has no FCI solver for state {state}",
            )
        try:
            ci_root = ci_frag[state]
        except (IndexError, TypeError) as err:
            raise ValueError(
                f"Fragment {ifrag} has no CI vector for state {state}",
            ) from err
        if ci_root is None:
            raise ValueError(
                f"Fragment {ifrag} CI vector for state {state} is missing",
            )
        fcisolvers.append(solvers[state])
        ci_state.append(ci_root)

    return fcisolvers, ci_state, ncas_sub, nelecas_sub


def make_one_casdm12_klas(klas, ci=None, state=0):
    """Build one kLAS state's active-space spin 1-RDMs and total 2-RDM.

    The returned matrices are expressed in the same Wannier active-orbital
    basis as the kLAS product-state CI.  No Wannier-to-k-point transformation
    is performed here.

    Parameters
    ----------
    klas : object
        A completed periodic LASCI or LASSCF object.
    ci : sequence, optional
        Fragment-major CI vectors.  The stored ``klas.ci`` is used by default.
    state : int, optional
        Rootspace index to select.

    Returns
    -------
    casdm1s : ndarray
        Spin-separated active-space 1-RDM with shape
        ``(2, ncastot, ncastot)``.
    casdm2 : ndarray
        Spin-summed active-space 2-RDM with shape ``(ncastot,) * 4``.
    """
    fcisolvers, ci_state, ncas_sub, nelecas_sub = _get_klas_rdm_context(
        klas, ci=ci, state=state,
    )
    solver = PBCProductStateFCISolver(
        fcisolvers,
        stdout=getattr(klas, "stdout", None),
        verbose=getattr(klas, "verbose", 0),
    )
    casdm1s = np.asarray(
        solver.make_rdm1s(ci_state, ncas_sub, nelecas_sub),
    )
    casdm2 = np.asarray(
        solver.make_rdm2(ci_state, ncas_sub, nelecas_sub),
    )

    ncastot = int(ncas_sub.sum())
    expected_dm1_shape = (2, ncastot, ncastot)
    expected_dm2_shape = (ncastot,) * 4
    if casdm1s.shape != expected_dm1_shape:
        raise ValueError(
            f"Expected kLAS spin 1-RDM shape {expected_dm1_shape}; "
            f"got {casdm1s.shape}",
        )
    if casdm2.shape != expected_dm2_shape:
        raise ValueError(
            f"Expected kLAS 2-RDM shape {expected_dm2_shape}; "
            f"got {casdm2.shape}",
        )
    if not np.all(np.isfinite(casdm1s)) or not np.all(np.isfinite(casdm2)):
        raise ValueError("kLAS density matrices must contain only finite values")
    return casdm1s, casdm2


def make_one_casdm1s_klas(klas, ci=None, state=0):
    """Return one kLAS state's spin-separated Wannier-basis active 1-RDMs."""
    return make_one_casdm12_klas(klas, ci=ci, state=state)[0]


def make_one_casdm2_klas(klas, ci=None, state=0):
    """Return one kLAS state's spin-summed Wannier-basis active 2-RDM."""
    return make_one_casdm12_klas(klas, ci=ci, state=state)[1]


def get_klas_mo_phase(klas, mo_coeff=None):
    """Return the Bloch-to-Wannier phase matching a kLAS active space.

    kLAS density matrices are defined by
    :func:`mrh.my_pyscf.pbc.util.wannier.get_wannier_orbs`.  Reusing a phase
    generated by a different periodic MC-SCF convention can silently reorder
    or rotate their Wannier indices.  This helper therefore applies the kLAS
    Wannier routine directly to the final active block of ``mo_coeff`` and
    validates the resulting square transformation.

    Parameters
    ----------
    klas : object
        A periodic LASCI or LASSCF object providing ``_scf``, ``kmesh``,
        ``ncore``, ``ncas``, and ``ncas_sub``.
    mo_coeff : ndarray, optional
        Block orbitals with shape ``(nkpts, nao, nmo)``.  The stored
        ``klas.mo_coeff`` is used by default.

    Returns
    -------
    mo_phase : ndarray
        Unitary transformation with shape
        ``(nkpts, ncas, nkpts * ncas)``.  Its last index uses the same Wannier
        ordering as the kLAS product-state density matrices.
    """
    if mo_coeff is None:
        mo_coeff = getattr(klas, "mo_coeff", None)
    if mo_coeff is None:
        raise ValueError("The kLAS object has no molecular orbitals")
    mo_coeff = np.asarray(mo_coeff)
    if mo_coeff.ndim != 3:
        raise ValueError("mo_coeff must have shape (nkpts, nao, nmo)")

    try:
        ncore = int(klas.ncore)
        ncas = int(klas.ncas)
    except (AttributeError, TypeError, ValueError) as err:
        raise ValueError("klas.ncore and klas.ncas must be integers") from err
    nkpts = mo_coeff.shape[0]
    if ncore < 0 or ncas <= 0 or ncore + ncas > mo_coeff.shape[2]:
        raise ValueError("ncore and ncas are incompatible with mo_coeff")

    kpts = np.asarray(getattr(klas, "kpts", ()))
    if kpts.shape != (nkpts, 3):
        raise ValueError(
            f"kLAS kpts must have shape ({nkpts}, 3); got {kpts.shape}",
        )
    kmesh = np.asarray(getattr(klas, "kmesh", ()), dtype=int)
    if kmesh.shape != (3,) or np.any(kmesh <= 0):
        raise ValueError("kmesh must contain three positive integers")
    if int(np.prod(kmesh)) != nkpts:
        raise ValueError("The kmesh product must equal the number of k-points")

    ncas_sub = np.asarray(getattr(klas, "ncas_sub", ()), dtype=int)
    ncastot = nkpts * ncas
    if ncas_sub.ndim != 1 or int(ncas_sub.sum()) != ncastot:
        raise ValueError(
            f"sum(ncas_sub) must equal nkpts * ncas = {ncastot}",
        )

    mo_active = np.ascontiguousarray(
        mo_coeff[:, :, ncore:ncore + ncas],
    )
    mo_phase = np.asarray(
        get_wannier_orbs(klas._scf, tuple(kmesh), mo_active)[-1],
        dtype=np.result_type(mo_coeff.dtype, np.complex128),
    )
    expected_shape = (nkpts, ncas, ncastot)
    if mo_phase.shape != expected_shape:
        raise ValueError(
            f"Expected kLAS mo_phase shape {expected_shape}; "
            f"got {mo_phase.shape}",
        )
    if not np.all(np.isfinite(mo_phase)):
        raise ValueError("kLAS mo_phase must contain only finite values")

    phase_matrix = mo_phase.reshape(ncastot, ncastot)
    identity = np.eye(ncastot, dtype=phase_matrix.dtype)
    if not (
            np.allclose(
                phase_matrix.conj().T @ phase_matrix,
                identity,
                atol=1e-8,
                rtol=1e-8,
            )
            and np.allclose(
                phase_matrix @ phase_matrix.conj().T,
                identity,
                atol=1e-8,
                rtol=1e-8,
            )):
        raise ValueError("stacked kLAS mo_phase must be unitary")
    return mo_phase
