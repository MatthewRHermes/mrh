from types import SimpleNamespace
from functools import partial

import numpy as np

from pyscf import lib
from pyscf.pbc.lib import kpts_helper
from pyscf.mcpdft import _dms

from mrh.my_pyscf.pbc.fci.addons import _unpack_nelec
from mrh.my_pyscf.pbc.mcscf.k2R import get_mo_coeff_k2R_wokmf
from mrh.my_pyscf.pbc.mcscf.mc1step import (
    _get_casdm2_kpts as _basis_transform_casdm2_kpts,
)
from mrh.my_pyscf.pbc.util.wannier import get_wannier_orbs

"""Reduced-density-matrix utilities for periodic MC-PDFT.

The kCASCI solvers store active-space RDMs in flattened Bloch-orbital order,
``k * ncas + orbital``.  This module builds those RDMs, extracts their
momentum-conserving blocks, transforms active 1-RDMs to AO densities, and
forms complex-orbital cumulants.  Charged-sector selection belongs to the
k-MC-PDFT driver; charged helpers here operate on an already selected result.
"""


def _mat_shape_check(mat, expected, label):
    # Instead of chekcing the shape inline and printing the error msg
    # I am using this fn to check the shape of the matrix and print the error msg
    mat = np.asarray(mat)
    matches = len(mat.shape) == len(expected) and all(
        want is None or have == want
        for have, want in zip(mat.shape, expected)
    )
    if not matches:
        raise ValueError(
            f"Expected {label} shape {expected}, got {mat.shape}",
        )
    return mat

def dm2_cumulant_complex(dm2, dm1s):
    """Build the two-body cumulant for a complex orbital basis."""
    dm1s = np.asarray(dm1s)
    return _dms.dm2_cumulant(dm2, dm1s.swapaxes(-1, -2))


def _get_kcas_rdm_context(mc, ci, state=0):
    """Resolve one neutral kCASCI state and its RDM arguments."""
    nkpts, ncas = mc.nkpts, mc.ncas
    target_k = int(mc.target_k) % nkpts
    fcisolver, ci, nelecas = _dms._get_fcisolver(mc, ci, state=state)

    spin = getattr(getattr(mc, "cell", None), "spin", None)
    nelecas = _unpack_nelec(nelecas, spin)

    ncastot = nkpts * ncas
    nelecastot = tuple(nkpts * value for value in nelecas)
    kwargs = {"nkpts": nkpts, "target_k": target_k}
    return fcisolver, ci, ncastot, nelecastot, kwargs


def _get_charged_kcas_rdm_context(mc, result, ci=None, state=0):
    """Resolve RDM arguments for an already selected charged kCAS result."""
    nkpts, ncas = mc.nkpts, mc.ncas
    target_k = int(result["target_k"]) % nkpts
    if ci is None: ci = result.get("ci")
    if ci is None:
        msg = (f"The charged KCASCI result for target_k={target_k} has no CI "
            "vector.",)
        raise ValueError(msg)

    fcisolver, ci, _ = _dms._get_fcisolver(mc, ci, state=state)

    nelecastot = result.get("nelecastot") or getattr(mc, "charged_nelecastot", None)
    if nelecastot is None:
        raise ValueError("No valid electron count found for the charged kCAS result.")
    ncastot = nkpts * ncas
    kwargs = {"nkpts": nkpts, "target_k": target_k}
    return fcisolver, ci, ncastot, nelecastot, kwargs


def _make_casdm1s(context, label):
    fcisolver, ci, ncastot, nelecastot, kwargs = context
    expected = (2, ncastot, ncastot)
    return _mat_shape_check(
        fcisolver.make_rdm1s(ci, ncastot, nelecastot, **kwargs),
        expected, f"spin-separated {label} 1-RDM",
    )


def _make_casdm2(context, label):
    fcisolver, ci, ncastot, nelecastot, kwargs = context
    try:
        _, casdm2 = fcisolver.make_rdm12(
            ci, ncastot, nelecastot, **kwargs,
        )
    except AttributeError:
        casdm2 = fcisolver.make_rdm2(ci, ncastot, nelecastot, **kwargs)
    expected = (ncastot,) * 4
    return _mat_shape_check(casdm2, expected, f"{label} 2-RDM")


def make_one_casdm1s_kcas(mc, ci, state=0):
    """Build one neutral kCASCI state's spin-separated active 1-RDM."""
    return _make_casdm1s(
        _get_kcas_rdm_context(mc, ci, state=state), "kCASCI",)


def make_one_casdm2_kcas(mc, ci, state=0):
    """Build one neutral kCASCI state's spin-summed active 2-RDM."""
    return _make_casdm2(
        _get_kcas_rdm_context(mc, ci, state=state), "kCASCI",)


def make_one_casdm1s_charged_kcas(mc, result, ci=None, state=0):
    """Build a selected charged kCASCI state's spin-separated 1-RDM."""
    return _make_casdm1s(
        _get_charged_kcas_rdm_context(mc, result, ci=ci, state=state),
        "charged KCASCI",)


def make_one_casdm2_charged_kcas(mc, result, ci=None, state=0):
    """Build a selected charged kCASCI state's spin-summed 2-RDM."""
    return _make_casdm2(
        _get_charged_kcas_rdm_context(mc, result, ci=ci, state=state),
        "charged KCASCI",)

def _check_forbidden_norm(total_sq, forbidden_sq, tolerance, label):
    if tolerance is None: return
    total = np.sqrt(max(0.0, float(total_sq)))
    forbidden = np.sqrt(max(0.0, float(forbidden_sq)))
    if forbidden > tolerance * max(1.0, total):
        raise ValueError(
            f"{label} contains momentum-forbidden blocks with norm "
            f"{forbidden:.3e} for momentum_tol={tolerance:.3e}",
        )


def casdm1s_to_kpts(casdm1s, nkpts, ncas, momentum_tol=1e-8):
    """Extract k-diagonal blocks from a flattened Bloch-basis 1-RDM."""
    ncastot = nkpts * ncas
    expected = (2, ncastot, ncastot)
    casdm1s = _mat_shape_check(
        casdm1s, expected, "spin-separated kCASCI 1-RDM",
    )

    full = casdm1s.reshape(2, nkpts, ncas, nkpts, ncas)
    blocks = np.zeros((2, nkpts, ncas, ncas), dtype=casdm1s.dtype,)
    forbidden_sq = 0.0
    for k1, k2 in np.ndindex(nkpts, nkpts):
        block = full[:, k1, :, k2, :]
        if k1 == k2: blocks[:, k1] = block
        else: forbidden_sq += np.vdot(block, block).real

    # Check the norm in forbidden blocks against the total norm of the 1-RDM.
    _check_forbidden_norm(
        np.vdot(casdm1s, casdm1s).real, forbidden_sq,
        momentum_tol, "kCASCI 1-RDM",)

    del full, forbidden_sq

    return blocks


def cascm2_to_kpts(cascm2, nkpts, ncas, kconserv, momentum_tol=1e-8):
    """Extract momentum-conserving blocks from a Bloch-basis cumulant."""
    kconserv = np.asarray(kconserv)
    ncastot = nkpts * ncas
    expected = (ncastot,) * 4
    cascm2 = _mat_shape_check(cascm2, expected, "kCASCI cumulant")

    shape = (nkpts, ncas, nkpts, ncas, nkpts, ncas, nkpts, ncas)
    full = cascm2.reshape(shape)
    blocks = np.zeros(
        (nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas),
        dtype=cascm2.dtype,
    )
    forbidden_sq = 0.0
    for k1, k2, k3, k4 in np.ndindex((nkpts,) * 4):
        block = full[k1, :, k2, :, k3, :, k4, :]
        if k4 == kconserv[k1, k2, k3]:
            blocks[k1, k2, k3] = block
        else:
            forbidden_sq += np.vdot(block, block).real

    # Check the norm in forbidden blocks against the total norm of the cumulant.
    _check_forbidden_norm(
        np.vdot(cascm2, cascm2).real, forbidden_sq,
        momentum_tol, "kCASCI cumulant",
    )
    return blocks


def make_kcas_rdms_kpts(casdm1s, casdm2, nkpts, ncas, kconserv,
                         momentum_tol=1e-8):
    """Convert dense kCASCI RDMs into periodic momentum blocks."""
    dm1s = casdm1s_to_kpts(casdm1s, nkpts, ncas, momentum_tol=momentum_tol,)
    cumulant = dm2_cumulant_complex(casdm2, casdm1s)
    cm2 = cascm2_to_kpts(
        cumulant, nkpts, ncas, kconserv, momentum_tol=momentum_tol,)
    return dm1s, cm2


def casdm1s_kpts_to_dm1s(obj, casdm1s_kpts, mo_coeff, ncore):
    """Transform k-resolved active 1-RDMs to spin-separated AO matrices."""

    del obj  # Retained for compatibility with the molecular helper's API.

    mo_coeff = _mat_shape_check(
        mo_coeff, (None, None, None), "mo_coeff",
    )

    nkpts = mo_coeff.shape[0]
    casdm1s_kpts = _mat_shape_check(
        casdm1s_kpts, (2, nkpts, None, None), "casdm1s_kpts",
    )
    ncas = casdm1s_kpts.shape[2]
    expected = (2, nkpts, ncas, ncas)
    casdm1s_kpts = _mat_shape_check(
        casdm1s_kpts, expected, "casdm1s_kpts",
    )

    ncore = int(ncore)

    if ncore < 0 or ncore + ncas > mo_coeff.shape[2]:
        raise ValueError("ncore and ncas are incompatible with mo_coeff")

    mo_core = mo_coeff[:, :, :ncore]
    mo_cas = mo_coeff[:, :, ncore:ncore + ncas]

    dm1s_cas = np.einsum("kap,skpq,kbq->skab",
                         mo_cas, casdm1s_kpts, mo_cas.conj(),
                         optimize=True,)
    dm1_core = np.einsum(
        "kai,kbi->kab", mo_core, mo_core.conj(), optimize=True,)

    return dm1s_cas + dm1_core[None]


def make_klas_mo_phase(kmf, kmesh, mo_coeff, ncore, ncas, ncas_sub=None):
    """Same as get_klas_mo_phase, but takes the pieces of the kLAS object.

    Args:
        kmf: Mean-field object (or any object with cell, kpts and get_ovlp).
        kmesh: K-point mesh.
        mo_coeff: Orbital coefficients, shape (nkpts, nao, nmo).
        ncore: Number of core orbitals per unit cell.
        ncas: Number of active orbitals per unit cell.
    Kwargs:
        ncas_sub: Active orbitals per fragment; if given, its sum must equal
            nkpts * ncas.
    Returns:
        mo_phase: Coefficients <Bloch(k, a) | Wannier(P)>, shape
            (nkpts, ncas, nkpts * ncas). Stacking (k, a) gives a unitary matrix.
    """
    mo_coeff = np.asarray(mo_coeff)
    if mo_coeff.ndim != 3:
        raise ValueError("mo_coeff must have shape (nkpts, nao, nmo)")

    nkpts, _, _ = mo_coeff.shape
    ncastot = nkpts * ncas
    if ncas_sub is not None and sum(ncas_sub) != ncastot:
        raise ValueError(f"sum(ncas_sub) must equal nkpts * ncas = {ncastot}")

    mo_active = np.ascontiguousarray(mo_coeff[:, :, ncore:ncore + ncas])
    mo_phase = np.asarray(
        get_wannier_orbs(kmf, kmesh, mo_active)[-1],
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
    expected_shape = (nkpts, nkpts, nkpts)
    if kconserv.shape != expected_shape:
        raise ValueError(
            f"kconserv shape must be {expected_shape}; got {kconserv.shape}"
        )

    casdm1s = np.asarray(casdm1s, dtype = mo_phase.dtype)
    casdm2 = np.asarray(casdm2, dtype = mo_phase.dtype)

    _mat_shape_check(casdm1s, (2, ncastot, ncastot), "casdm1s")
    _mat_shape_check(casdm2, (ncastot,) * 4, "casdm2")

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


def _get_wannier_mo_phase(ot, mo_coeff, ncore, ncas):
    """Build mo_phase with the generic k-to-R transformation.

    args:
        ot: otfnalperiodic_kpts object
            On-top functional; provides cell, kpts and kmesh.
        mo_coeff: np.ndarray (nkpts, nao, nmo)
            MO coefficients at each k-point.
        ncore: int
            Number of core orbitals per unit cell.
        ncas: int
            Number of active orbitals per unit cell.

    returns:
        mo_phase: np.ndarray (nkpts, ncas, nkpts*ncas)
            Coefficients <Bloch(k, a) | active orbital of the supercell>.
    """
    if getattr(ot, 'kmesh', None) is None:
        raise ValueError("kmesh is required for Wannier RDMs")
    return get_mo_coeff_k2R_wokmf(
        ot.cell, mo_coeff, ncore, ncas, ot.kpts, kmesh=ot.kmesh,
    )[-1]


def _get_klas_mo_phase(ot, mo_coeff, ncore, ncas):
    """Build mo_phase with the kLAS orbital convention.

    Uses make_klas_mo_phase. If ot has no _scf attribute, a
    minimal stand-in for the mean-field object is built from ot.cell and
    ot.kpts. Arguments and return value are the same as
    _get_wannier_mo_phase.
    """
    if getattr(ot, 'kmesh', None) is None:
        raise ValueError("kmesh is required for kLAS RDMs")
    kmf = getattr(ot, '_scf', None)
    if kmf is None:
        kmf = SimpleNamespace(
            cell=ot.cell, kpts=ot.kpts,
            get_ovlp=lambda kpts: ot.cell.pbc_intor('int1e_ovlp', kpts=kpts),
        )
    return make_klas_mo_phase(
        kmf, ot.kmesh, mo_coeff, ncore, ncas,
        ncas_sub=getattr(ot, 'ncas_sub', None),
    )


def _prepare_wannier_rdms(ot, casdm1s, casdm2, mo_coeff, ncore, get_mo_phase):
    """Transform Wannier-basis RDMs to momentum blocks.

    The Wannier-to-k-space phase is built by the get_mo_phase callable, which
    is the only thing that differs between the 'wannier' and 'klas'
    representations. The 2-RDM is converted to its cumulant and transformed
    block by block for every (k1, k2, k3), with k4 fixed by momentum
    conservation, in make_klas_rdms_kpts.

    args:
        ot: otfnalperiodic_kpts object
            On-top functional; provides cell, kpts, kmesh and, if set,
            kconserv.
        casdm1s: np.ndarray (2, ncas*nkpts, ncas*nkpts)
            Spin-separated active 1-RDM in the Wannier basis.
        casdm2: np.ndarray (ncas*nkpts,)*4
            Active 2-RDM in the Wannier basis.
        mo_coeff: np.ndarray (nkpts, nao, nmo)
            MO coefficients at each k-point.
        ncore: int
            Number of core orbitals per unit cell.
        get_mo_phase: callable
            get_mo_phase(ot, mo_coeff, ncore, ncas) -> mo_phase of shape
            (nkpts, ncas, nkpts*ncas).

    returns:
        casdm1s_kpts: np.ndarray (2, nkpts, ncas, ncas)
            Spin-separated active 1-RDM at each k-point.
        cascm2_kpts: np.ndarray (nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas)
            Active 2-RDM cumulant for the k-point triplet (k1, k2, k3);
            k4 is fixed by kconserv.
        kconserv: np.ndarray (nkpts, nkpts, nkpts)
            Momentum-conservation table for the k-points.
    """
    mo_coeff = np.asarray(mo_coeff)
    casdm2 = np.asarray(casdm2)
    if mo_coeff.ndim != 3:
        raise ValueError("mo_coeff must have shape (nkpts, nao, nmo)")
    if casdm2.ndim != 4 or len(set(casdm2.shape)) != 1:
        raise ValueError("casdm2 must have shape (ncas * nkpts,) * 4")
    nkpts = mo_coeff.shape[0]
    if casdm2.shape[0] % nkpts:
        raise ValueError("The active RDM size must be divisible by nkpts")
    ncas = casdm2.shape[0] // nkpts
    if ncore < 0 or ncore + ncas > mo_coeff.shape[2]:
        raise ValueError("ncore and ncas are incompatible with mo_coeff")

    mo_phase = get_mo_phase(ot, mo_coeff, ncore, ncas)
    kconserv = getattr(ot, 'kconserv', None)
    if kconserv is None:
        kconserv = kpts_helper.get_kconserv(ot.cell, ot.kpts)
    casdm1s_kpts, cascm2_kpts = make_klas_rdms_kpts(
        casdm1s, casdm2, mo_phase, kconserv,
    )
    return casdm1s_kpts, cascm2_kpts, kconserv


def _prepare_bloch_rdms(ot, casdm1s, casdm2, mo_coeff, momentum_tol):
    """Extract momentum blocks from flattened Bloch-basis RDMs.

    The RDMs are already in the Bloch (k-space) basis, so no orbital
    transformation is done; only the momentum-conserving blocks are
    extracted with make_kcas_rdms_kpts.

    args:
        ot: otfnalperiodic_kpts object
            On-top functional; provides cell, kpts and, if set,
            kconserv.
        casdm1s: np.ndarray (2, ncas*nkpts, ncas*nkpts)
            Spin-separated active 1-RDM in the Bloch basis.
        casdm2: np.ndarray (ncas*nkpts,)*4
            Active 2-RDM in the Bloch basis.
        mo_coeff: np.ndarray (nkpts, nao, nmo)
            MO coefficients at each k-point.
        momentum_tol: float
            Tolerance used to identify momentum-conserving blocks.

    returns:
        casdm1s_kpts: np.ndarray (2, nkpts, ncas, ncas)
            Spin-separated active 1-RDM at each k-point.
        cascm2_kpts: np.ndarray (nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas)
            Active 2-RDM cumulant for the k-point triplet (k1, k2, k3);
            k4 is fixed by kconserv.
        kconserv: np.ndarray (nkpts, nkpts, nkpts)
            Momentum-conservation table for the k-points.
    """
    mo_coeff = np.asarray(mo_coeff)
    casdm2 = np.asarray(casdm2)
    if mo_coeff.ndim != 3:
        raise ValueError("mo_coeff must have shape (nkpts, nao, nmo)")
    if casdm2.ndim != 4 or len(set(casdm2.shape)) != 1:
        raise ValueError("casdm2 must have shape (ncas * nkpts,) * 4")

    nkpts = mo_coeff.shape[0]
    ncastot = casdm2.shape[0]
    if ncastot % nkpts:
        raise ValueError("The active RDM size must be divisible by nkpts")
    ncas = ncastot // nkpts
    kconserv = getattr(ot, 'kconserv', None)
    if kconserv is None:
        kconserv = kpts_helper.get_kconserv(ot.cell, ot.kpts)
    casdm1s_kpts, cascm2_kpts = make_kcas_rdms_kpts(
        casdm1s, casdm2, nkpts, ncas, kconserv,
        momentum_tol=momentum_tol,
    )
    return casdm1s_kpts, cascm2_kpts, kconserv


def _prepare_kpts_rdms(ot, casdm1s, casdm2, mo_coeff, ncore,
                       representation, momentum_tol):
    """Prepare active RDMs for the shared k-point evaluator.

    Dispatches to the converter that matches the basis of the input RDMs.

    args:
        ot: otfnalperiodic_kpts object
            On-top functional; provides cell, kpts and, if set,
            kconserv.
        casdm1s: np.ndarray (2, ncas*nkpts, ncas*nkpts)
            Spin-separated active 1-RDM in the basis given by representation.
        casdm2: np.ndarray (ncas*nkpts,)*4
            Active 2-RDM in the basis given by representation.
        mo_coeff: np.ndarray (nkpts, nao, nmo)
            MO coefficients at each k-point.
        ncore: int
            Number of core orbitals per unit cell.
        representation: str
            Basis of the input RDMs: 'wannier', 'bloch' or 'klas'.
        momentum_tol: float
            Tolerance for momentum-conserving blocks ('bloch' only).

    returns:
        Same as the selected converter: (casdm1s_kpts, cascm2_kpts,
        kconserv) with shapes (2, nkpts, ncas, ncas),
        (nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas) and
        (nkpts, nkpts, nkpts).
    """
    if representation == 'wannier':
        prepare = partial(
            _prepare_wannier_rdms, get_mo_phase=_get_wannier_mo_phase,
        )
        return prepare(ot, casdm1s, casdm2, mo_coeff, ncore)
    if representation == 'bloch':
        return _prepare_bloch_rdms(
            ot, casdm1s, casdm2, mo_coeff, momentum_tol,
        )
    if representation == 'klas':
        prepare = partial(
            _prepare_wannier_rdms, get_mo_phase=_get_klas_mo_phase,
        )
        return prepare(ot, casdm1s, casdm2, mo_coeff, ncore)
    raise ValueError(f"Unknown RDM representation {representation!r}")
