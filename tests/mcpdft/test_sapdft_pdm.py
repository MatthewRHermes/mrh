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
def get_h2o(iroots=3):
    weights = [1/iroots]*iroots
    mol = gto.M(atom = geom_h2o, basis = '6-31g',
             symmetry='c2v', output='/dev/null', verbose=0)
    mf = scf.RHF(mol).run()
    mc = mcpdft.CASSCF(mf,'tPBE', 4, 4, grids_level=1)
    mc = mc.state_average_(weights)
    mc.kernel(mf.mo_coeff)
    return mc

class KnownValues(unittest.TestCase):
    
    def test_h2o_sa3_tpbe_631g(self):
        dm_ref = np.array(\
            [[-1.67550633e-16,  2.23632807e+00,  9.65318153e-17],
            [ -2.18697668e-15, -5.49340995e-01,  1.50687836e-16],
            [-1.41095270e-15, -6.49403912e-01,  1.47911868e-16]])

        iroots=3
        mc = get_h2o(iroots)
        for i in range(iroots):
            with self.subTest (i=i):
                dm_test = mc.dip_moment(unit='Debye', origin="Coord_center",state=i)
                for dmt,dmr in zip(dm_test,dm_ref[i]):
                    self.assertAlmostEqual(dmt,dmr)
        
if __name__ == "__main__":
    print("Test for SA-PDFT permanent dipole moments")
    unittest.main()
