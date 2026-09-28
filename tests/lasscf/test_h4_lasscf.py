import unittest
from pyscf import gto, scf, lib
from mrh.my_pyscf.mcscf import lasscf_async as asyn
from mrh.my_pyscf.mcscf import lasscf_sync_o0 as syn

def setUpModule():
    global mol, mf
    xyz = '''H 0.0 0.0 0.0
             H 1.0 0.0 0.0
             H 0.2 3.9 0.1
             H 1.159166 4.1 -0.1'''
    mol = gto.M (atom = xyz, basis = '6-31g',
                 output='/dev/null',
                 verbose=0)
    mf = scf.RHF (mol).run ()

def tearDownModule():
    global mol, mf
    mf.stdout.close ()
    del mol, mf

class KnownValues (unittest.TestCase):

    def test_state_average_async (self):
        frag_atom_list = ((0,),(1,),(2,),(3,))
        las_asyn = asyn.LASSCF (mf, (2,2,2,2), ((1,0),(0,1),(1,0),(0,1)), spin_sub=(2,2,2,2))
        mo_loc = las_asyn.set_fragments_(frag_atom_list, mf.mo_coeff)
        las_asyn.state_average_(weights=[1.0/6,]*6,
                                spins=[[1,-1,1,-1],[-1,1,1,-1],[1,-1,-1,1],[-1,1,-1,1],[1,1,-1,-1],[-1,-1,1,1]],
                                smults=[[2,2,2,2],]*6,
                                charges=[[0,0,0,0],]*6)
        las_asyn.kernel (mo_loc)
        self.assertTrue (las_asyn.converged)
        
        las_syn = syn.LASSCF (mf, (2,2,2,2), ((1,0),(0,1),(1,0),(0,1)), spin_sub=(2,2,2,2))
        mo_loc = las_syn.localize_init_guess (frag_atom_list, mf.mo_coeff)
        las_syn.state_average_(weights=[1.0/6,]*6,
                               spins=[[1,-1,1,-1],[-1,1,1,-1],[1,-1,-1,1],[-1,1,-1,1],[1,1,-1,-1],[-1,-1,1,1]],
                               smults=[[2,2,2,2],]*6,
                               charges=[[0,0,0,0],]*6)
        las_syn.kernel (mo_loc)
        self.assertTrue (las_syn.converged)
        
        self.assertAlmostEqual (las_syn.e_tot,
                                las_asyn.e_tot,
                                7)

        self.assertAlmostEqual (lib.fp (las_syn.e_states),
                                lib.fp (las_asyn.e_states),
                                4)

if __name__ == "__main__":
    print ("Full Tests for H4")
    unittest.main()
