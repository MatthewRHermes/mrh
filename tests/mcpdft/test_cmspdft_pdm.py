import numpy as np
from pyscf import gto, scf, mcscf
from mrh.my_pyscf import mcpdft
import unittest
from mrh.my_pyscf.fci import csf_solver

geom_h2o='''
O  0.00000000   0.08111156   0.00000000
H  0.78620605   0.66349738   0.00000000
H -0.78620605   0.66349738   0.00000000
'''
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
def get_h2o():
    weights = [1/3]*3
    mol = gto.M(atom = geom_h2o, basis = '6-31g',
             symmetry='c2v', output='/dev/null', verbose=0)
    mf = scf.RHF(mol).run()
    mc = mcscf.CASSCF(mf, 4,4)
    mc = mc.state_average_(weights)
    mc.kernel(mf.mo_coeff)

    mcms = mcpdft.CASSCF(mf,'tPBE', 4, 4, grids_level=1)
    mcms = mcms.multi_state(weights = weights)
    mcms.kernel(mc.mo_coeff, mc.ci)
    return mcms

def get_furan_cation(iroots=3):# A2,B2, and A2 states
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
    
    def test_h2o_cms3_tpbe_631g(self):
        dm_ref = np.array(\
            [[-1.26839490e-09,2.34877543e+00,-2.57099161e-11],
            [  5.64381079e-16,-5.49340995e-01,1.50676478e-16],
            [  1.46930730e-09,-1.30914327e-01,1.16235053e-11]])

        mc = get_h2o()
        for i in range(3):
            with self.subTest (i=i):
                dm_test = mc.dip_moment(unit='Debye', origin="Coord_center",state=i)
                for dmt,dmr in zip(dm_test,dm_ref[i]):
                    self.assertAlmostEqual(dmt,dmr)
        
    def test_furan_cation_cms3_tblyp_sto3g(self):
        dm_ref = np.array(\
            [[1.53083932e-15, -1.01474042e+00, -1.04204761e+00],
            [ 1.04025912e-15, -1.10039329e+00, -1.12987135e+00],
            [ 1.64503146e-15, -3.95041004e-01, -4.05738740e-01]])

        iroots=3
        mc = get_furan_cation(iroots)
        for i in range(iroots):
            with self.subTest (i=i):
                dm_test = mc.dip_moment(unit='Debye', origin="charge_center",state=i)
                for dmt,dmr in zip(dm_test,dm_ref[i]):
                    self.assertAlmostEqual(dmt,dmr)

if __name__ == "__main__":
    print("Test for CMS-PDFT permanent dipole moments")
    unittest.main()
