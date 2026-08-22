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
