#!/bin/bash
import copy
import unittest
import numpy as np
from scipy.linalg import expm

from pyscf.pbc import scf
from pyscf.pbc import gto as pgto

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.fci import csf_solver
from mrh.my_pyscf.pbc.mcscf.k2R import get_mo_coeff_k2R


'''
This is the unit test for the k-CASSCF orbital hessian operator.
In this unit test, I would check whether the hessian operator behaves as the first order changes in 
the gradients with respect to the orbital rotation parameters.

Test-1: In the limit of single determinant, the hessian operator should reduce to 
the k-RHF orbital hessian.
Test-2: For more than SD case, I will compare the Taylor expansion residual error (as described above) with
as a function of the orbital rotation step size.
'''

def get_cell():
    cell = pgto.Cell(
        a=np.diag([3.0, 10.0, 10.0]),
        atom="Ne 0.0 0.0 0.0",
        basis="6-31G",
        unit="Angstrom",
        ke_cutoff=50,
        precision=1e-12,
        verbose=0,
    )
    return cell

def get_cell2():
    cell = pgto.Cell(
        a=np.diag([3.0, 10.0, 10.0]),
        atom="N 0.0 0.0 0.0; N 0.0 0.0 1.1",
        basis="STO-3G",
        unit="Angstrom",
        ke_cutoff=50,
        precision=1e-12,
        verbose=0,)
    return cell

def scramble_within_subspaces(mc, mo_coeff, rng_seed=65):
    rng = np.random.default_rng(rng_seed)
    mo_new = np.array(mo_coeff, copy=True)
    nmo = mo_coeff[0].shape[1]
    for k in range(mc.nkpts):
        perm = rng.permutation(np.arange(nmo))
        mo_new[k] = mo_new[k][:, perm]
    return mo_new

def rotate_mos_kpts(kmc, mo_coeff, x, step=1.0):
    x = np.asarray(x).reshape(-1)
    nkpts = kmc.nkpts
    nvar_k = x.size // nkpts
    x = x.reshape(nkpts, nvar_k)
    mo_new = np.array(mo_coeff, copy=True)
    for k in range(nkpts):
        dr_k = kmc.unpack_uniq_var(step * x[k])
        kappa_k = 0.5 * (dr_k - dr_k.conj().T)
        mo_new[k] = mo_new[k] @ expm(kappa_k)
    return mo_new

def get_casdm12_k(mc, ci):
    ncastot = mc.ncas * mc.nkpts
    nelecastot = (mc.nelecas[0] * mc.nkpts, mc.nelecas[1] * mc.nkpts)
    return mc.fcisolver.make_rdm12(ci, ncastot, nelecastot)

def get_g_hop_frozen_ci_k(mc, mo_coeff, ci):
    eris = mc.ao2mo(mo_coeff)
    mo_phase = get_mo_coeff_k2R(mc._scf, mo_coeff.copy(), mc.ncore, mc.ncas)[-1]
    casdm1, casdm2 = get_casdm12_k(mc, ci)
    gorb = mc.gen_g_hop(mo_coeff, mo_phase, 1, casdm1, casdm2, eris)[0]
    return gorb.ravel()

def get_hess_op(mc, mo_coeff, ci):
    eris = mc.ao2mo(mo_coeff)
    mo_phase = get_mo_coeff_k2R(mc._scf, mo_coeff.copy(), mc.ncore, mc.ncas)[-1]
    casdm1, casdm2 = get_casdm12_k(mc, ci)
    hop = mc.gen_g_hop(mo_coeff, mo_phase, 1, casdm1, casdm2, eris)[2]
    return hop

def krhf_hess_op(kmf, mo_ref):
    mo_ref = mo_ref.copy()
    mo_occ = np.asarray(kmf.mo_occ)
    hop = kmf.gen_g_hop(mo_coeff=mo_ref, mo_occ=mo_occ)[1]
    return hop

def hop_taylor_residual(mc, mo_ref, ci0, x0, scales):
    g0 = get_g_hop_frozen_ci_k(mc, mo_ref.copy(), ci0)
    hop0 = get_hess_op(mc, mo_ref.copy(), ci0)
    x0 = np.asarray(x0).reshape(-1)
    nx0 = np.linalg.norm(x0)
    assert nx0 > 1e-30, "Reference direction x0 has zero norm."
    x0 = x0 / nx0 # Normalizing it.
    results = {"scale": [], "norm_dg": [], "norm_hx": [], 
               "norm_resid": [], "eps_tilde": [], "cos_angle": []}

    for s in scales:
        x = s * x0
        mo_rot = rotate_mos_kpts(mc, mo_ref.copy(), x)
        gx = get_g_hop_frozen_ci_k(mc, mo_rot.copy(), ci0)

        dg = gx - g0
        hx = np.asarray(hop0(x)).reshape(-1)
        resid = dg - hx

        norm_dg = np.linalg.norm(dg)
        norm_hx = np.linalg.norm(hx)
        norm_resid = np.linalg.norm(resid)
        eps = np.nan if norm_dg < 1e-30 else norm_resid / norm_dg

        if norm_dg > 0 and norm_hx > 0:
            cosang = np.real(np.vdot(dg, hx)) / (norm_dg * norm_hx)
        else:
            cosang = np.nan

        results["scale"].append(s)
        results["norm_dg"].append(norm_dg)
        results["norm_hx"].append(norm_hx)
        results["norm_resid"].append(norm_resid)
        results["eps_tilde"].append(eps)
        results["cos_angle"].append(cosang)

    for k in results:
        results[k] = np.asarray(results[k])

    return results

class KnownValues(unittest.TestCase):
    def taylor_residual_check(
        self,
        results,
        max_eps_by_scale=None,
        min_cos_by_scale=None,
        monotonic_after=None,
    ):
        scales = results["scale"]
        eps = results["eps_tilde"]
        cosang = results["cos_angle"]

        self.assertEqual(scales.shape, eps.shape)
        self.assertEqual(scales.shape, cosang.shape)

        self.assertFalse(np.any(np.isnan(eps)), 
                         msg=f"NaN found in eps_tilde: {eps}")
        self.assertFalse(np.any(np.isnan(cosang)), 
                         msg=f"NaN found in cos_angle: {cosang}")

        if max_eps_by_scale is not None:
            for s_max, eps_max in max_eps_by_scale:
                mask = scales <= s_max
                if np.any(mask):
                    this_eps = eps[mask]
                    self.assertTrue(np.all(this_eps < eps_max))

        if min_cos_by_scale is not None:
            for s_max, cos_min in min_cos_by_scale:
                mask = scales <= s_max
                if np.any(mask):
                    this_cos = cosang[mask]
                    self.assertTrue(np.all(this_cos > cos_min))

        if monotonic_after is not None:
            mask = scales <= monotonic_after
            idx = np.where(mask)[0]
            if idx.size >= 2:
                eps_small = eps[idx]
                self.assertTrue(np.all(eps_small[1:] <= eps_small[:-1] + 1e-12))

    def test_hop_kRHF_limit(self):
        cell = get_cell2()
        cell.build()
        kmesh1D = [2, 1, 1]
        kpts = cell.make_kpts(kmesh1D, wrap_around=True)
        kmf = scf.KRHF(cell, exxdiv=None, 
                       kpts=kpts).density_fit(auxbasis="def2-svp-jkfit").newton()
        kmf.max_cycle = 0
        kmf.exxdiv = None
        kmf.kernel()

        mo_ref = np.array(kmf.mo_coeff, copy=True)

        kmc2 = mcscf.CASSCF(kmf, 7, (7, 7))
        kmc2.max_cycle_macro = 0
        kmc2.fcisolver = csf_solver(cell, smult=1)

        mo_ref = scramble_within_subspaces(kmc2, mo_ref, rng_seed=7)
        kmc2.kernel(mo_ref.copy())

        kmc2.mo_coeff = mo_ref.copy()
        mo_ref = np.array(kmc2.mo_coeff, copy=True)
        ci0 = copy.deepcopy(kmc2.ci)

        hop_krhf = krhf_hess_op(kmf, mo_ref.copy())
        hop_kmc2 = get_hess_op(kmc2, mo_ref.copy(), ci0.copy())

        gorb_kmf = kmf.get_grad(mo_ref.copy(), kmf.mo_occ.copy())
        gorb_kmc2 = kmc2.get_grad(mo_ref.copy())

        hx_kmf = np.asarray(hop_krhf(gorb_kmf).ravel())
        hx_kmc2 = np.asarray(hop_kmc2(gorb_kmc2).ravel())

        self.assertTrue(np.allclose(gorb_kmf, gorb_kmc2, atol=1e-8, rtol=1e-6))
        self.assertTrue(np.allclose(hx_kmf, hx_kmc2, atol=1e-8, rtol=1e-6))

    def test_hop_taylor_residual_kcasscf_frozen_ci(self):
        cell = get_cell()
        cell.build()
        kmesh1D = [3, 1, 1]
        kpts = cell.make_kpts(kmesh1D, wrap_around=True)
        kmf = scf.KRHF(cell, exxdiv=None, 
                       kpts=kpts).density_fit(auxbasis="def2-svp-jkfit").newton()
        kmf.max_cycle = 0
        kmf.exxdiv = None
        kmf.kernel()

        mo_ref = np.array(kmf.mo_coeff, copy=True)

        kmc2 = mcscf.CASSCF(kmf, 2, (1, 1))
        kmc2.max_cycle_macro = 0
        kmc2.fcisolver = csf_solver(cell, smult=1)

        mo_ref = scramble_within_subspaces(kmc2, mo_ref, rng_seed=7)
        kmc2.kernel(mo_ref.copy())

        kmc2.mo_coeff = mo_ref.copy()
        mo_ref = np.array(kmc2.mo_coeff, copy=True)
        ci0 = copy.deepcopy(kmc2.ci)

        x0 = get_g_hop_frozen_ci_k(kmc2, mo_ref.copy(), ci0)
        x0_norm = np.linalg.norm(x0)
        self.assertGreater(x0_norm, 1e-30, 
                           msg="Reference direction x0 has near-zero norm.")
        x0 /= x0_norm

        scales = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7])

        results = hop_taylor_residual(kmc2, mo_ref, ci0, x0, scales)

        self.taylor_residual_check(
            results,
            max_eps_by_scale=[
                (1e-2, 5e-2),
                (1e-4, 5e-3),
                (1e-6, 1e-3),
            ],
            min_cos_by_scale=[
                (1e-2, 0.99),
                (1e-4, 0.999),
            ],
            monotonic_after=1e-2,
        )

if __name__ == "__main__":
    unittest.main()