import numpy as np
from pyscf import gto, scf, mcscf
from mrh.my_pyscf import mcpdft
import unittest

def get_h2o():
    weights = [1/3]*3
    mol = gto.M(atom = '''O  0.00000000   0.08111156   0.00000000
             H  0.78620605   0.66349738   0.00000000
             H -0.78620605   0.66349738   0.00000000''', basis = '6-31g',
             symmetry='c2v', output='/dev/null', verbose=0)
    mf = scf.RHF(mol).run()
    mc = mcscf.CASSCF(mf, 4,4)
    mc = mc.state_average_(weights)
    mc.kernel(mf.mo_coeff)

    mcms = mcpdft.CASSCF(mf,'tPBE', 4, 4, grids_level=1)
    mcms = mcms.multi_state(weights = weights)
    mcms.kernel(mc.mo_coeff, mc.ci)
    return mcms

class KnownValues(unittest.TestCase):
    
    def test_h2o_cms2tpbe_sto3g(self):
        dm_ref = np.array([[-1.26839490e-09,2.34877543e+00,-2.57099161e-11],
            [5.64381079e-16,-5.49340995e-01,1.50676478e-16],
            [1.46930730e-09,-1.30914327e-01,1.16235053e-11]])

        mc = get_h2o()
        for i in range(3):
            with self.subTest (i=i):
                dm_test = mc.dip_moment(unit='Debye', state=i)
                for dmt,dmr in zip(dm_test,dm_ref[i]):
                    self.assertAlmostEqual(dmt,dmr)


if __name__ == "__main__":
    print("Full Test for electrical CMS-PDFT dipole moments")
    unittest.main()
