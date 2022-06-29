import numpy as np
from pyscf import gto, scf, mcscf
from mrh.my_pyscf import mcpdft
import unittest
from mrh.my_pyscf.fci import csf_solver

geom_furan= '''
C        0.000000000     -0.965551055     -2.020010585
C        0.000000000     -1.993824223     -1.018526668
C        0.000000000     -1.352073201      0.181141565
O        0.000000000      0.000000000      0.000000000
C        0.000000000      0.216762264     -1.346821565
H        0.000000000     -1.094564216     -3.092622941
H        0.000000000     -3.062658055     -1.175803180
H        0.000000000     -1.688293885      1.206105691
H        0.000000000      1.250242874     -1.655874372
'''
def get_furan(iroots=3):
    weights = [1/iroots]*iroots
    mol = gto.M(atom = geom_furan, basis = 'sto-3g',
             symmetry='C2v', output='/dev/null', verbose=0)
    mf = scf.RHF(mol).run()
    mc = mcpdft.CASSCF(mf,'tPBE', 5, 6, grids_level=1)
    mc.fcisolver = csf_solver(mol, smult=1, symm='A1')
    mc = mc.multi_state(weights, 'cms')
    mc.max_cycle_macro = 200
    mo = mcscf.sort_mo(mc, mf.mo_coeff, [12,17,18,19,20])
    mc.kernel(mo)
    return mc

def get_furan_cation(iroots=3):
    weights = [1/iroots]*iroots
    mol = gto.M(atom = geom_furan, basis = 'sto-3g', charge=1, spin=1,
             symmetry=False, output='/dev/null', verbose=0)
    mf = scf.RHF(mol).run()
    mc = mcpdft.CASSCF(mf,'tBLYP', 5, 5, grids_level=1)
    mc.fcisolver = csf_solver(mol, smult=2)
    mc = mc.multi_state(weights, 'cms')
    mc.max_cycle_macro = 200
    mo = mcscf.sort_mo(mc, mf.mo_coeff, [12,17,18,19,20])
    mc.kernel(mo)
    return mc

class KnownValues(unittest.TestCase):
    
    def test_furan_cms3_tpbe_sto3g(self):
        tdm_ref = np.array(\
        [[-9.69627146e-18, -2.04665383e-01, -2.10140092e-01],
        [ -1.18108406e-17, -6.83355318e-01, -7.01634766e-01],
        [ -1.03002864e-17, -7.03590398e-01, -7.22411131e-01]])

        iroots=3
        k=0
        mc = get_furan(iroots)
        for i in range(iroots):
            for j in range(i):
                with self.subTest (k=k):
                    tdm_test = mc.trans_moment(\
                        unit='Debye', origin="mass_center",state=[i,j])
                    for tdmt,tdmr in zip(tdm_test,tdm_ref[k]):
                        try:
                            self.assertAlmostEqual(tdmt,tdmr)
                        except:
                            self.assertAlmostEqual(-tdmt,tdmr)
                k += 1

    def test_furan_cation_cms2_tblyp_sto3g(self):
        tdm_ref = np.array(\
        [[-3.99323562e-17,  3.58096881e-01, -3.48749335e-01],
        [  1.10330560e-16, -3.42447674e-01,  3.33516674e-01],
        [ -1.41464772e-16,  1.84049078e-01,  1.88940159e-01]])

        iroots=3
        k=0
        mc = get_furan_cation(iroots)
        for i in range(iroots):
            for j in range(i):
                with self.subTest (k=k):
                    tdm_test = mc.trans_moment(\
                        unit='AU', origin="Charge_center",state=[i,j])
                    for tdmt,tdmr in zip(tdm_test,tdm_ref[k]):
                        try:
                            self.assertAlmostEqual(tdmt,tdmr)
                        except:
                            self.assertAlmostEqual(-tdmt,tdmr)
                k += 1  

if __name__ == "__main__":
    print("Test for CMS-PDFT transition dipole moments")
    unittest.main()
