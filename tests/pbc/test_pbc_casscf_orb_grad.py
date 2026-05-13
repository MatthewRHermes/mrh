import unittest

import numpy as np
import scipy
import copy
from scipy.linalg import expm

from pyscf import lib
from pyscf.pbc import scf
from pyscf.pbc import gto as pgto
from functools import reduce

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.fci import csf_solver
from mrh.my_pyscf.pbc.mcscf import avas
from mrh.my_pyscf.pbc.mcscf.mc1step import block_diag_to_kblocks
from mrh.my_pyscf.pbc.mcscf.k2R import get_mo_coeff_k2R

# Unit test: Orbital gradient for KHF 
# Test-1: In the limit of the single determinant the KHF orbital gradient would be equals to
# the k-CASSCF orbital gradient. Note, the SD should constitute the entire doubly occupied space.
# Test-2: Orbital Gradient:
# Test-3: Grad_update function:

def get_cell():
    cell = pgto.Cell(
        a=np.diag([3.0, 10.0, 10.0]),
        atom="Be 0.0 0.0 0.0",
        basis="6-31G",
        unit="Angstrom",
        max_memory=100000,
        ke_cutoff=50,
        precision=1e-12,
        verbose=0,
    )
    return cell

def scramble_within_subspaces(mc, mo_coeff, rng_seed=65):
    rng = np.random.default_rng(rng_seed)
    mo_new = np.array(mo_coeff, copy=True)
    nmo = mo_coeff[0].shape[1]
    for k in range(mc.nkpts):
        perm = rng.permutation(np.arange(nmo))
        mo_new[k] = mo_new[k][:, perm]
    return mo_new

def generate_kappa(nkpts, nmo, scale=1e-3, seed=12):
    rng = np.random.default_rng(seed)
    kappa_k = np.zeros((nkpts, nmo, nmo), dtype=complex)
    for k in range(nkpts):
        x = rng.standard_normal((nmo, nmo)) + 1j * rng.standard_normal((nmo, nmo))
        kappa = x - x.conj().T
        kappa *= scale
        kappa_k[k] = kappa
    return kappa_k

def rotate_mos(mo_coeff, kappa, scale=1.0):
    nkpts = len(mo_coeff)
    mo_rot = np.array([c.copy() for c in mo_coeff])
    kappa = np.asarray(kappa)
    for k in range(nkpts):
        U = expm(scale * kappa[k])
        mo_rot[k] = mo_rot[k] @ U
    return mo_rot

def get_casdm12_k(mc, ci):
    ncastot = mc.ncas * mc.nkpts
    nelecastot = (mc.nelecas[0] * mc.nkpts, mc.nelecas[1] * mc.nkpts)
    return mc.fcisolver.make_rdm12(ci, ncastot, nelecastot)

def get_gorb_k(mc, mo_coeff, ci):
    eris = mc.ao2mo(mo_coeff)
    mo_phase = get_mo_coeff_k2R(mc._scf, np.array(mo_coeff, copy=True), mc.ncore, mc.ncas)[-1]
    casdm1, casdm2 = get_casdm12_k(mc, ci)
    gorb = mc.gen_g_hop(mo_coeff, mo_phase, 1, casdm1, casdm2, eris)[0]
    return np.asarray(gorb).reshape(-1)

def compute_mcscf_energy_at_mo(mc_ref, mo_coeff):
    mf = mc_ref._scf
    mc_temp = mcscf.CASCI(mf, mc_ref.ncas, mc_ref.nelecas)
    mc_temp.fcisolver = csf_solver(mf.cell, smult=1)

    mo_trial = [c.copy().astype(dtype=complex) for c in mo_coeff]

    e = mc_temp.kernel(mo_trial)[0]
    return e

def unpack_uniq_var(mc, v):
    v = np.asarray(v)
    nkpts = mc.nkpts
    nmo = mc.mo_coeff[0].shape[1]
    dtype = np.result_type(mc.mo_coeff[0].dtype, v.dtype)

    idx = mc.uniq_var_indices(nmo, mc.ncore, mc.ncas, mc.frozen)
    nuniq = int(np.count_nonzero(idx))

    if v.size not in (nuniq, nkpts * nuniq):
        raise ValueError(f"Unexpected packed size {v.size}; expected {nuniq} or {nkpts*nuniq}")

    def _unpack_one(v1):
        mat = np.zeros((nmo, nmo), dtype=dtype)
        mat[idx] = v1
        return mat - mat.conj().T

    if v.size == nuniq:
        return _unpack_one(v)

    mats = np.zeros((nkpts, nmo, nmo), dtype=dtype)
    for k in range(nkpts):
        p0 = k * nuniq
        p1 = (k + 1) * nuniq
        mats[k] = _unpack_one(v[p0:p1])
    return mats

def random_packed_direction(mc, seed=7):
    nkpts = mc.nkpts
    nmo = mc.mo_coeff[0].shape[1]
    idx = mc.uniq_var_indices(nmo, mc.ncore, mc.ncas, mc.frozen)
    nuniq = int(np.count_nonzero(idx))

    rng = np.random.default_rng(seed)
    x = rng.standard_normal(nkpts * nuniq) + 1j * rng.standard_normal(nkpts * nuniq)
    x /= np.linalg.norm(x)
    return x

def compute_grad_error(mc, step_sizes, x0, grad_packed, mo_ref, use_total_energy=False):
    nkpts = mc.nkpts
    x0 = np.asarray(x0, dtype=complex)
    x0 = x0 / np.linalg.norm(x0)

    grad_ref = np.asarray(grad_packed, dtype=complex).copy()
    if use_total_energy:
        grad_ref = nkpts * grad_ref

    E0 = compute_mcscf_energy_at_mo(mc, mo_ref)
    if use_total_energy:
        E0 *= nkpts

    results = {
        "scale": [],
        "dE": [],
        "g_dot_x": [],
        "abs_epsilon": [],
    }

    for s in step_sizes:
        X = s * x0
        kappa = unpack_uniq_var(mc, X)
        mo_rot = rotate_mos(mo_ref, kappa, scale=1.0)

        E = compute_mcscf_energy_at_mo(mc, mo_rot)
        if use_total_energy:
            E *= nkpts

        dE = E - E0
        g_dot_x = (2.0 / nkpts) * np.real(np.vdot(grad_ref, X))
        resid = dE - g_dot_x
        eps = np.nan if abs(dE) < 1e-30 else abs(resid / dE)

        results["scale"].append(s)
        results["dE"].append(dE)
        results["g_dot_x"].append(g_dot_x)
        results["abs_epsilon"].append(eps)

    for key in results:
        results[key] = np.asarray(results[key])

    return results

class KnownValues(unittest.TestCase):
    def taylor_residual_check(self, results, max_eps_by_scale=None,
                              monotonic_after=None):
        scales = results["scale"]
        eps = results["abs_epsilon"]

        self.assertEqual(scales.shape, eps.shape)
        self.assertFalse(np.any(np.isnan(eps)),
                         msg=f"NaN found in abs_epsilon: {eps}")

        if max_eps_by_scale is not None:
            for s_max, eps_max in max_eps_by_scale:
                mask = scales <= s_max
                if np.any(mask):
                    this_eps = eps[mask]
                    self.assertTrue(np.all(this_eps < eps_max))

        if monotonic_after is not None:
            mask = scales <= monotonic_after
            idx = np.where(mask)[0]
            if idx.size >= 2:
                eps_small = eps[idx]
                self.assertTrue(np.all(eps_small[1:] <= eps_small[:-1] + 1e-12))

    def test_kmf_kmc_orb_grad(self):
        cell = pgto.Cell()
        cell.a = np.diag([5.0, 10.0, 10.0])
        cell.atom = '''
        Be 0.0 0.0 0.0
        Be 2.0 0.0 0.0
        '''
        cell.basis = '6-31G'
        cell.unit = 'Angstrom'
        cell.ke_cutoff = 100
        cell.precision = 1e-12
        cell.verbose = 0
        cell.build()

        kmesh1D = [3, 1, 1]

        kpts = cell.make_kpts(kmesh1D, wrap_around=True)

        kmf = scf.KRHF(cell, kpts=kpts).density_fit(auxbasis='def2-svp-jkfit')
        kmf.exxdiv = None
        kmf.conv_tol = 1e-1
        kmf.kernel()

        mo_ref = kmf.mo_coeff.copy()

        kmc = mcscf.CASSCF(kmf, 4, (4,4))
        kmc.max_cycle_macro = 0
        kmc.fcisolver = csf_solver(cell, smult=1)
        kmc.kernel(mo_ref)

        # Now compute the gradients
        kmf_orb_grad = kmf.get_grad(mo_ref, kmf.mo_occ)
        kmc_orb_grad = kmc.get_grad(mo_coeff=mo_ref)

        for a, b in zip(kmf_orb_grad, kmc_orb_grad):
            self.assertAlmostEqual(a, b, places=7)

    def Atest_kmc_gorb_taylor_residual(self):
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

        g0 = get_gorb_k(kmc2, mo_ref.copy(), ci0)
        x0 = g0 #random_packed_direction(kmc2, seed=7)
        x0_norm = np.linalg.norm(x0)
        self.assertGreater(x0_norm, 1e-30,
                        msg="Reference direction x0 has near-zero norm.")
        x0 /= x0_norm

        scales = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7])

        results = compute_grad_error(kmc2, scales, x0, g0, mo_ref, use_total_energy=False)

        print(results["abs_epsilon"])
        self.taylor_residual_check(
            results,
            max_eps_by_scale=[
                (1e-2, 5e-2),
                (1e-4, 5e-3),
                (1e-6, 1e-3),
            ],
            monotonic_after=1e-2,
        )

    def Atest_kmf_kmc_orb_grad_update(self):
        cell = pgto.Cell()
        cell.a = np.diag([3.0, 10.0, 10.0])
        cell.atom = '''
        Be 0.0 0.0 0.0
        '''
        cell.basis = '6-31G'
        cell.unit = 'Angstrom'
        cell.ke_cutoff = 50
        cell.precision = 1e-12
        cell.verbose = 0
        cell.build()


        kmesh1D = [3, 1, 1]

        kpts = cell.make_kpts(kmesh1D, wrap_around=True)

        kmf = scf.KRHF(cell, kpts=kpts).density_fit(auxbasis='def2-svp-jkfit')
        kmf.max_cycle=10
        kmf.exxdiv = None
        kmf.kernel()

        mo_coeff = np.array(kmf.mo_coeff).copy()
        
        mo_ref = np.array(mo_coeff).copy()

        kmc2 = mcscf.CASSCF(kmf, 1, (1,1))
        kmc2.max_cycle_macro = 0
        kmc2.fcisolver = csf_solver(cell, smult=1)
        kmc2.kernel(mo_coeff)

        # Compute the gradient and the update function
        grad = kmc2.get_grad(mo_ref.copy())
        grad_update = kmc2.get_grad_update(mo_ref.copy())
        # Compute the updated gradient (i) very small rotation (1e-10) such that U=I and 
        # (ii) using finite size rotation (1e-3) to check the consistency of the grad_update function.
        def _generate_grad_and_grad_updates(scale, mo_ref, grad_update):
            nkpts = kmc2.nkpts
            kappa_k = generate_kappa(nkpts, mo_ref[0].shape[1], scale, seed=12)
            u_k = np.array([scipy.linalg.expm(kappa_k[k]) 
                            for k in range(nkpts)])
            umat = scipy.linalg.block_diag(*[scipy.linalg.expm(kappa_k[k]) 
                                             for k in range(nkpts)])
            
            mo_rot = np.array([mo_ref[k] @ u_k[k] 
                               for k in range(nkpts)])

            kmc_rot = mcscf.CASSCF(kmf, 1, (1, 1))
            kmc_rot.max_cycle_macro = 0
            kmc_rot.fcisolver = csf_solver(cell, smult=1)
            e_rot = kmc_rot.kernel(mo_rot)[0]

            # direct gradient at rotated orbitals
            grad_direct = kmc_rot.get_grad(mo_rot)
            grad_from_update = grad_update(umat, kmc2.ci)
            return grad_direct, grad_from_update, e_rot
        
        def _compare_lists(list1, list2, places=7):
            for a, b in zip(list1, list2):
                self.assertAlmostEqual(a, b, places=places)

        grad_direct_1, grad_update_1, e_rot_1 = _generate_grad_and_grad_updates(1e-10, mo_ref, grad_update)
        grad_direct_2, grad_update_2, e_rot_2 = _generate_grad_and_grad_updates(1e-3, mo_ref, grad_update)

        # First vanilla check for energiesi
        self.assertAlmostEqual(kmc2.e_tot, e_rot_1, places=7)

        # For very small rotation, the gradients should be almost equal
        _compare_lists(grad, grad_direct_1, places=5)
        _compare_lists(grad, grad_update_1, places=5)

        # For finite rotation, the gradients won't be exactly equal recomputed gradientsbut should be close.
        # This similar test on the molecular CASSCF is agreeing to the 4th decimal place.
        _compare_lists(grad_direct_2, grad_update_2, places=4)

    
if __name__ == "__main__":
    # Orbital gradient test for k-CASSCF.
    unittest.main()