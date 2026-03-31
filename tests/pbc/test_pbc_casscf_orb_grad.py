import unittest

import numpy as np
import scipy
from scipy.linalg import expm

from pyscf import lib
from pyscf.pbc import scf
from pyscf.pbc import gto as pgto


from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.fci import csf_solver
from mrh.my_pyscf.pbc.mcscf import avas


# Unit test: Orbital gradient for KHF 
# Test-1: In the limit of the single determinant the KHF orbital gradient would be equals to
# the k-CASSCF orbital gradient. Note, the SD should constitute the entire doubly occupied space.
# Test-2: Orbital Gradient: # TODO (Error function test suggested by Matt)
# Test-3: Grad_update function:

def generate_kappa(nkpts, nmo, scale=1e-3, seed=12):
    rng = np.random.default_rng(seed)
    kappa_k = np.zeros((nkpts, nmo, nmo), dtype=complex)
    for k in range(nkpts):
        x = rng.standard_normal((nmo, nmo)) + 1j * rng.standard_normal((nmo, nmo))
        kappa = x - x.conj().T
        kappa *= scale
        kappa_k[k] = kappa
    return kappa_k
class KnownValues(unittest.TestCase):
    
    def test_kmf_kmc_orb_grad(self):
        cell = pgto.Cell()
        cell.a = np.diag([5.0, 10.0, 10.0])
        cell.atom = '''
        Be 0.0 0.0 0.0
        Be 2.0 0.0 0.0
        '''
        cell.basis = '6-31G'
        cell.unit = 'Angstrom'
        cell.max_memory = 100000
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
    
    def test_kmf_kmc_orb_grad_update(self):
        cell = pgto.Cell()
        cell.a = np.diag([3.0, 10.0, 10.0])
        cell.atom = '''
        Be 0.0 0.0 0.0
        '''
        cell.basis = '6-31G'
        cell.unit = 'Angstrom'
        cell.max_memory = 100000
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