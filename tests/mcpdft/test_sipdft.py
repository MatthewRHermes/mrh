import numpy as np
from scipy import linalg
from pyscf import gto, scf, df, mcscf, lib
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.grad.sipdft import sipdft_heff_response, sipdft_heff_HellmanFeynman
from mrh.my_pyscf.df.grad import dfsacasscf
import unittest, math

#def tearDownModule():
#    global mol_nosymm, mol_symm, mc_list, si
#    mol_nosymm.stdout.close ()
#    mol_symm.stdout.close ()
#    del mol_nosymm, mol_symm, mc_list, si

class KnownValues(unittest.TestCase):

    def test_scanner (self):
        def get_lih (r):
            mol = gto.M (atom='Li 0 0 0\nH {} 0 0'.format (r), basis='sto3g',
                         output='test.{}.log'.format (r), verbose=5)
            mf = scf.RHF (mol).run ()
            mc = mcpdft.CASSCF (mf, 'ftLDA,VWN3', 2, 2, grids_level=1)
            mc.fix_spin_(ss=0)
            mc = mc.state_interaction ([0.5,0.5], 'cms').run (conv_tol=1e-8)
            return mol, mc.run ()
        mol1, mc1 = get_lih (1.5)
        mol2, mc2 = get_lih (1.55)
        mc2 = mc2.as_scanner () 
        mc2 (mol1)
        self.assertTrue(mc1.converged)
        self.assertTrue(mc2.converged)
        for state in 0,1:
            e1 = mc1.e_states[state]
            e2 = mc2.e_states[state]
            self.assertAlmostEqual (e1, e2, 6)
    

if __name__ == "__main__":
    print("Full Tests for SI-PDFT energy API")
    unittest.main()






