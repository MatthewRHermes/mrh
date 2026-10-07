import numpy as np

from pyscf.pbc.lib import kpts_helper

from mrh.my_pyscf.pbc.mcscf.productstate import PBCProductStateFCISolver
from mrh.my_pyscf.pbc.mcscf.mc1step import (
    _get_casdm2_kpts as _basis_transform_casdm2_kpts,
)
from mrh.my_pyscf.pbc.mcpdft._dms import dm2_cumulant_complex
from mrh.my_pyscf.pbc.util.wannier import get_wannier_orbs
from mrh.my_pyscf.pbc.mcscf.klasscf import _check_shape

# Author: Bhavnesh Jangid

"""Build Wannier-basis kLAS RDMs and transform them to k-point PDFT blocks."""


def _get_klas_rdm_context(klas, ci=None, state=0):
    """Select fragment solvers and CI vectors for one kLAS rootspace.

    Args:
        klas: Periodic LASCI or LASSCF object containing fragment states.
    Kwargs:
        ci: Fragment CI vectors indexed as ci[ifrag][state]; defaults to klas.ci.
        state: Rootspace index; defaults to 0.
    Returns:
        fcisolvers: List of fragment FCI solvers for the selected rootspace.
        ci_state: List of fragment CI vectors for the selected rootspace.
        ncas_sub: Active orbital counts, shape (nfrags,).
        nelecas_sub: Fragment alpha/beta electron counts, shape (nfrags, 2).
    """
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
    """Build Wannier-basis RDMs, including interfragment direct and exchange terms.

    Args:
        klas: Periodic LASCI or LASSCF object containing fragment states.
    Kwargs:
        ci: Fragment CI vectors indexed as ci[ifrag][state]; defaults to klas.ci.
        state: Rootspace index; defaults to 0.
    Returns:
        casdm1s: Alpha/beta 1-RDMs, shape (2, ncastot, ncastot), using <p^+ q>.
        casdm2: Spin-summed 2-RDM, shape (ncastot,) * 4.
        Both use fragment orbital order, with ncastot = sum(klas.ncas_sub).
    """
    
    fcisolvers, ci_state, ncas_sub, nelecas_sub = _get_klas_rdm_context(
        klas, ci=ci, state=state,
    )
    
    solver = PBCProductStateFCISolver(
        fcisolvers,
        stdout=getattr(klas, "stdout", None),
        verbose=getattr(klas, "verbose", 0),
    )

    casdm1s = np.asarray(solver.make_rdm1s(ci_state, ncas_sub, nelecas_sub),)
    casdm2 = np.asarray(solver.make_rdm2(ci_state, ncas_sub, nelecas_sub),)

    ncastot = int(ncas_sub.sum())
    _check_shape(casdm1s, (2, ncastot, ncastot), "casdm1s")
    _check_shape(casdm2, (ncastot,) * 4, "casdm2")
    return casdm1s, casdm2


def make_one_casdm1s_klas(klas, ci=None, state=0):
    """Return Wannier-basis spin 1-RDMs via make_one_casdm12_klas.

    Args:
        klas: Periodic LASCI or LASSCF object containing fragment states.
    Kwargs:
        ci: Fragment CI vectors indexed as ci[ifrag][state]; defaults to klas.ci.
        state: Rootspace index; defaults to 0.
    Returns:
        casdm1s: Alpha/beta 1-RDMs, shape (2, ncastot, ncastot),
            where ncastot = sum(klas.ncas_sub).
    """
    return make_one_casdm12_klas(klas, ci=ci, state=state)[0]


def make_one_casdm2_klas(klas, ci=None, state=0):
    """Return the Wannier-basis spin-summed 2-RDM via make_one_casdm12_klas.

    Args:
        klas: Periodic LASCI or LASSCF object containing fragment states.
    Kwargs:
        ci: Fragment CI vectors indexed as ci[ifrag][state]; defaults to klas.ci.
        state: Rootspace index; defaults to 0.
    Returns:
        casdm2: Full product-state 2-RDM, shape (ncastot,) * 4,
            where ncastot = sum(klas.ncas_sub).
    """
    return make_one_casdm12_klas(klas, ci=ci, state=state)[1]


def get_klas_mo_phase(klas, mo_coeff=None):
    """Build and validate the orbital transformation matching the kLAS Wannier basis.

    Args:
        klas: Periodic LASCI or LASSCF object defining the orbitals and k mesh.
    Kwargs:
        mo_coeff: Orbital coefficients, shape (nkpts, nao, nmo);
            defaults to klas.mo_coeff. Only the active block is transformed.
    Returns:
        mo_phase: Coefficients <Bloch(k, a) | Wannier(P)>, shape
            (nkpts, ncas, nkpts * ncas). Stacking (k, a) gives a unitary matrix.
    """
    if mo_coeff is None:
        mo_coeff = klas.mo_coeff
    mo_coeff = np.asarray(mo_coeff)
    if mo_coeff.ndim != 3:
        raise ValueError("mo_coeff must have shape (nkpts, nao, nmo)")

    nkpts, _, _ = mo_coeff.shape
    ncore, ncas = klas.ncore, klas.ncas
    ncastot = nkpts * ncas
    if sum(klas.ncas_sub) != ncastot:
        raise ValueError(f"sum(ncas_sub) must equal nkpts * ncas = {ncastot}")

    mo_active = np.ascontiguousarray(mo_coeff[:, :, ncore:ncore + ncas])
    mo_phase = np.asarray(
        get_wannier_orbs(klas._scf, klas.kmesh, mo_active)[-1],
        dtype=np.result_type(mo_coeff.dtype, np.complex128),
    )
    
    if mo_phase.shape != (nkpts, ncas, ncastot):
        msg = (
            f"Expected kLAS mo_phase shape {(nkpts, ncas, ncastot)}; "
            f"got {mo_phase.shape}"
        )
        raise ValueError(msg)
    
    phase_matrix = mo_phase.reshape(ncastot, ncastot)
    if not np.allclose(
            phase_matrix.conj().T @ phase_matrix, np.eye(ncastot),
            atol=1e-8, rtol=1e-8):
        raise ValueError("stacked kLAS mo_phase must be unitary")
    return mo_phase


def make_klas_rdms_kpts(casdm1s, casdm2, mo_phase, kconserv):
    """Transform Wannier RDMs to k-point 1-RDM and two-body cumulant blocks.

    Args:
        casdm1s: Wannier alpha/beta 1-RDMs, shape (2, ncastot, ncastot).
        casdm2: Wannier spin-summed 2-RDM, shape (ncastot,) * 4.
        mo_phase: Coefficients <Bloch(k, a) | Wannier(P)>, shape
            (nkpts, ncas, ncastot), with ncastot = nkpts * ncas.
        kconserv: Integer lookup, shape (nkpts, nkpts, nkpts), giving
            k4 = kconserv[k1, k2, k3] for each momentum-conserving tuple.
    Returns:
        casdm1s_kpts: Spin 1-RDM blocks, shape (2, nkpts, ncas, ncas).
        cascm2_kpts: Cumulant blocks, shape
            (nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas); k4 is implicit.
    """
    mo_phase = np.asarray(mo_phase)
    if mo_phase.ndim != 3:
        msg = "mo_phase must have shape (nkpts, ncas, nkpts * ncas)"
        raise ValueError(msg)
    
    nkpts, ncas, ncastot = mo_phase.shape
    
    if ncastot != nkpts * ncas:
        msg = "mo_phase must map a square stacked active-orbital space"
        raise ValueError(msg)
    
    kconserv = np.asarray(kconserv)

    casdm1s = np.asarray(casdm1s, dtype = mo_phase.dtype)
    casdm2 = np.asarray(casdm2, dtype = mo_phase.dtype)

    _check_shape(casdm1s, (2, ncastot, ncastot), "casdm1s")
    _check_shape(casdm2, (ncastot,) * 4, "casdm2")

    casdm1s_kpts = np.einsum(
        "kap,spq,kbq->skab",
        mo_phase,
        casdm1s,
        mo_phase.conj(),
        optimize=True,
    )


    cascm2 = dm2_cumulant_complex(casdm2, casdm1s)
    dtype = np.result_type(cascm2.dtype, mo_phase.dtype)
    
    cascm2_kpts = np.empty(
        (nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas),
        dtype=dtype,
    )

    for k1, k2, k3 in kpts_helper.loop_kkk(nkpts):
        k4 = kconserv[k1, k2, k3]
        cascm2_kpts[k1, k2, k3] = _basis_transform_casdm2_kpts(
                cascm2, mo_phase, (k1, k2, k3, k4),)
    
    return casdm1s_kpts, cascm2_kpts
