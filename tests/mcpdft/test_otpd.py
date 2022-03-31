import numpy as np
from scipy import linalg
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf.mcpdft.otpd import get_ontop_pair_density
from mrh.my_pyscf.mcpdft.pdft_feff import EotOrbitalHessianOperator
from mrh.my_pyscf.mcpdft.pdft_feff import vector_error
import unittest

h2 = scf.RHF (gto.M (atom = 'H 0 0 0; H 1.2 0 0', basis = '6-31g', 
    output='/dev/null')).run ()
lih = scf.RHF (gto.M (atom = 'Li 0 0 0; H 1.2 0 0', basis = 'sto-3g',
    output='/dev/null')).run ()

def get_dm2_ao (mc, mo_coeff, casdm1, casdm2):
    i, ncas = mc.ncore, mc.ncas
    j = i + ncas
    mo_occ = mo_coeff[:,:j]
    dm1 = 2*np.eye (j)
    dm1[i:j,i:j] = casdm1
    dm2 = np.multiply.outer (dm1, dm1)
    dm2 -= 0.5*np.multiply.outer (dm1, dm1).transpose (0,3,2,1)
    dm2[i:j,i:j,i:j,i:j] = casdm2
    return np.einsum ('pqrs,ip,jq,kr,ls->ijkl', dm2, mo_occ, mo_occ,
                      mo_occ, mo_occ)

def get_rho_ref (dm1s, ao):
    rho = np.einsum ('sjk,caj,ak->sca', dm1s, ao[:4], ao[0])
    rho[:,1:4] += np.einsum ('sjk,cak,aj->sca', dm1s, ao[1:4], ao[0])
    return rho 

def get_Pi_ref (dm2, ao):
    nderiv, ngrid, nao = ao.shape
    Pi = np.zeros ((5,ngrid))
    Pi[:4]   = np.einsum ('ijkl,cai,aj,ak,al->ca', dm2,
                          ao[:4], ao[0], ao[0], ao[0]) / 2
    Pi[1:4] += np.einsum ('ijkl,caj,ai,ak,al->ca', dm2,
                          ao[1:4], ao[0], ao[0], ao[0]) / 2
    Pi[1:4] += np.einsum ('ijkl,cak,ai,aj,al->ca', dm2,
                          ao[1:4], ao[0], ao[0], ao[0]) / 2
    Pi[1:4] += np.einsum ('ijkl,cal,ai,aj,ak->ca', dm2,
                          ao[1:4], ao[0], ao[0], ao[0]) / 2
    X, Y, Z, XX, YY, ZZ = 1,2,3,4,7,9
    Pi[4]  = np.einsum ('ijkl,ai,aj,ak,al->a', dm2,
                        ao[XX], ao[0], ao[0], ao[0]) / 2
    Pi[4] += np.einsum ('ijkl,ai,aj,ak,al->a', dm2,
                        ao[YY], ao[0], ao[0], ao[0]) / 2
    Pi[4] += np.einsum ('ijkl,ai,aj,ak,al->a', dm2,
                        ao[ZZ], ao[0], ao[0], ao[0]) / 2
    Pi[4] += np.einsum ('ijkl,ai,aj,ak,al->a', dm2,
                        ao[X], ao[X], ao[0], ao[0]) / 2
    Pi[4] += np.einsum ('ijkl,ai,aj,ak,al->a', dm2,
                        ao[Y], ao[Y], ao[0], ao[0]) / 2
    Pi[4] += np.einsum ('ijkl,ai,aj,ak,al->a', dm2,
                        ao[Z], ao[Z], ao[0], ao[0]) / 2
    Pi[4] -= np.einsum ('ijkl,ai,aj,ak,al->a', dm2,
                        ao[X], ao[0], ao[X], ao[0]) / 2
    Pi[4] -= np.einsum ('ijkl,ai,aj,ak,al->a', dm2,
                        ao[Y], ao[0], ao[Y], ao[0]) / 2
    Pi[4] -= np.einsum ('ijkl,ai,aj,ak,al->a', dm2,
                        ao[Z], ao[0], ao[Z], ao[0]) / 2
    Pi[4] -= np.einsum ('ijkl,ai,aj,ak,al->a', dm2,
                        ao[X], ao[0], ao[0], ao[X]) / 2
    Pi[4] -= np.einsum ('ijkl,ai,aj,ak,al->a', dm2,
                        ao[Y], ao[0], ao[0], ao[Y]) / 2
    Pi[4] -= np.einsum ('ijkl,ai,aj,ak,al->a', dm2,
                        ao[Z], ao[0], ao[0], ao[Z]) / 2
    return Pi

def tearDownModule():
    global h2, lih
    h2.mol.stdout.close ()
    lih.mol.stdout.close ()
    del h2, lih

class KnownValues(unittest.TestCase):

    def test_de_d2e (self):
        for mol, mf in zip (('H2', 'LiH'), (h2, lih)):
            for state, nel in zip (('Singlet', 'Triplet'), (2, (2,0))):
                mc = mcpdft.CASSCF (mf, 'tLDA,VWN3', 2, nel, grids_level=1).run ()
                dm1s = np.array (mc.make_rdm1s ())
                casdm1s, casdm2s = mc.fcisolver.make_rdm12s (mc.ci, mc.ncas, mc.nelecas)
                casdm1 = casdm1s[0] + casdm1s[1]
                casdm2 = casdm2s[0] + casdm2s[1] + casdm2s[1].transpose (2,3,0,1) + casdm2s[2]
                cascm2 = casdm2 - np.multiply.outer (casdm1, casdm1)
                cascm2 += np.multiply.outer (casdm1s[0], casdm1s[0]).transpose (0,3,2,1)
                cascm2 += np.multiply.outer (casdm1s[1], casdm1s[1]).transpose (0,3,2,1)
                mo_cas = mc.mo_coeff[:,mc.ncore:][:,:mc.ncas]
                nao, ncas = mo_cas.shape
                with self.subTest (mol=mol, state=state):
                    ot, ni = mc.otfnal, mc.otfnal._numint
                    make_rho = tuple (ni._gen_rho_evaluator (ot.mol, dm1s[i], 1) for i in range (2))
                    dm2_ao = get_dm2_ao (mc, mc.mo_coeff, casdm1, casdm2)
                    for ao, mask, weight, coords in ni.block_loop (ot.mol, ot.grids, nao, 2, 2000):
                        rho = np.array ([m[0] (0, ao, mask, 'MGGA') for m in make_rho])
                        Pi_test = get_ontop_pair_density (
                            ot, rho, ao, dm1s, cascm2, mo_cas, deriv=2,
                            non0tab=mask)
                        Pi_ref = get_Pi_ref (dm2_ao, ao)
                        self.assertAlmostEqual (lib.fp (Pi_test), lib.fp (Pi_ref), 10)
                    

if __name__ == "__main__":
    print("Full Tests for MC-PDFT on-top pair density construction")
    unittest.main()






