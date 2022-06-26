import numpy as np
from scipy import linalg
from pyscf import gto, scf, df
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.prop.dip_moment.mspdft import ElectricDipole as CMSED
import unittest

def get_h2o():
    mol = gto.M(atom = '''O  0.00000000   0.08111156   0.00000000
             H  0.78620605   0.66349738   0.00000000
             H -0.78620605   0.66349738   0.00000000''', basis = 'sto3g',
             output='/dev/null', verbose=0)
    mf = scf.RHF(mol).run()
    mc = mcpdft.CASSCF(mf,'tPBE', 4, 4, grids_level=1)
    mc.fcisolver = csf_solver(mol, smult=1)
    mc = mc.multi_state([0.5,0.5], 'cms').run(mo=mf.mo_coeff, conv_tol=1e-8)
    return mc

class KnownValues(unittest.TestCase):
    
    def test_h2o_cms2tpbe_sto3g(self):
        dm_ref = np.array([[1.10016978e-01,1.65443708e+00,-4.62578013e-10],
            [-4.36406963e-01,-2.39398833e-03,9.54500757e-11]])

        mc = get_h2o()
        for i in range(2):
            with self.subTest (i=i):
                dm_test = mc.dip_moment(unit='Debye', state=i)
                for dmt,dmr in zip(dm_test,dm_ref[i]):
                    self.assertAlmostEqual(dmt,dmr)


if __name__ == "__main__":
    print("Full Test for electrical CMS-PDFT dipole moments")
    unittest.main()
