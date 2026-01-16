import unittest
from pyscf import gto, scf, lib
from mrh.my_pyscf.mcscf import lasscf_async as asyn
from mrh.my_pyscf.mcscf import lasscf_sync_o0 as syn

def setUpModule():
    global mol, mf
    xyz = '''H 0.0 0.0 0.0
             H 1.0 0.0 0.0'''
    mol = gto.M (atom = xyz, basis = '6-31g',
                 output='/dev/null',
                 verbose=0,
                 charge=-1, spin=1)
    mf = scf.RHF (mol).run ()

def tearDownModule():
    global mol, mf
    mf.stdout.close ()
    del mol, mf

class KnownValues (unittest.TestCase):

    def test_state_average_charge_hop_async (self):
        las_syn = syn.LASSCF (mf, (2,2), ((1,1),(1,0)), spin_sub=(1,2))
        mo_loc = las_syn.localize_init_guess (((0,),(1,)), mf.mo_coeff)
        las_syn.state_average_(weights=[.5,]*2,
                               spins=[[0,1],[1,0]],
                               smults=[[1,2],[2,1]],
                               charges=[[0,0],[1,-1]])
        las_syn.kernel (mo_loc)
        self.assertTrue (las_syn.converged)
        
        las_asyn = asyn.LASSCF (mf, (2,2), ((1,1),(1,0)), spin_sub=(1,2))
        mo_loc = las_asyn.set_fragments_(((0,),(1,)), mf.mo_coeff)
        las_asyn.state_average_(weights=[.5,]*2,
                                spins=[[0,1],[1,0]],
                                smults=[[1,2],[2,1]],
                                charges=[[0,0],[1,-1]])
        las_asyn.kernel (mo_loc)
        self.assertTrue (las_asyn.converged)

        self.assertAlmostEqual (lib.fp (las_syn.e_states),
                                lib.fp (las_asyn.e_states),
                                8)

if __name__ == "__main__":
    print ("Full Tests for H2-")
    unittest.main()
