import numpy as np

from pyscf import lib
from pyscf.mcpdft import _dms

from mrh.my_pyscf.pbc.fci.addons import _unpack_nelec

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
