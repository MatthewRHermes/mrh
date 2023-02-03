import unittest
import numpy as np
from pyscf import gto, scf, lib, mcscf
from me2n2_struct import structure as struct
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.mcscf.soc_int import compute_hso
from mrh.my_pyscf.mcscf.lassi_op_o0 import si_soc
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.mcscf.lassi import make_stdm12s, roots_make_rdm12s

def setUpModule():
    global molh2o, mfh2o, molme2n2, mfme2n2, lasme2n2
    molh2o = gto.M (atom="""
        O  0.000000  0.000000  0.000000
        H  0.758602  0.000000  0.504284
        H  0.758602  0.000000  -0.504284
    """, basis='631g',symmetry=True,
    output='test_lassi_soc.log',
    verbose=lib.logger.DEBUG)
    mfh2o = scf.RHF (molh2o).run ()
   
    molme2n2 = struct (3.0, '6-31g')
    molme2n2.output = 'test_lassi_soc2.log'
    molme2n2.verbose = lib.logger.DEBUG
    molme2n2.build ()
    mfme2n2 = scf.RHF (molme2n2).run ()
    lasme2n2 = LASSCF (mfme2n2, (4,4), (4,4), spin_sub=(1,1))
    lasme2n2.state_average_(weights=[1.0/5.0,]*5,
        spins=[[0,0],[0,0],[2,-2],[-2,2],[2,2]],
        smults=[[1,1],[3,3],[3,3],[3,3],[3,3]]).run()

def tearDownModule():
    global molh2o, mfh2o, molme2n2, mfme2n2, lasme2n2
    molh2o.stdout.close()
    molme2n2.stdout.close()
    del molh2o, mfh2o, molme2n2, mfme2n2, lasme2n2

class KnownValues (unittest.TestCase):

    def test_soc_int (self):
        # Pre-calculated atomic densities used in the OpenMolcas version of AMFI
        h2o_dm = np.zeros ((13,13))
        h2o_dm[0,0] = h2o_dm[1,1] = 2
        h2o_dm[3,3] = h2o_dm[4,4] = h2o_dm[5,5] = 4/3

        # Obtained from OpenMolcas v22.02
        int_ref = np.array ([0.0000000185242348, 0.0000393310222742, 0.0000393310222742, 0.0005295974407740]) 
        
        amfi_int = compute_hso (molh2o, h2o_dm, amfi=True)
        amfi_int = amfi_int[2][amfi_int[2] > 0]
        amfi_int = np.sort (amfi_int.imag)
        self.assertAlmostEqual (lib.fp (amfi_int), lib.fp (int_ref), 8)

    def test_soc_1frag (self):
        with lib.temporary_env (mfh2o.mol, charge=2):
            mc = mcscf.CASSCF (mfh2o, 8, 4).set (conv_tol=1e-12)
            mc.fcisolver = csf_solver (mfh2o.mol, smult=3).set (wfnsym='A1')
            mc.kernel ()
            # The result is very sensitive to orbital basis, so I optimize orbitals
            # tightly using CASSCF, which is a more stable implementation
            las = LASSCF (mfh2o, (8,), (4,), spin_sub=(3,), wfnsym_sub=('A1',))
            las.mo_coeff = mc.mo_coeff
            las.state_average_(weights=[1/4,]*4,
                               spins=[[2,],[0,],[-2,],[0]],
                               smults=[[3,],[3,],[3,],[1]],
                               wfnsyms=(([['B1',],]*3)+[['A1',],]))
            las.lasci ()
            e_roots, si = las.lassi (opt=0, soc=True, break_symmetry=True)
        # TODO: either validate this number or replace it with a more meaningful test.
        # I discovered by accident that the charge=+2 A1 singlet and B1 triplet of this water
        # molecule are close enough in energy for a usable splitting here if you use the
        # orbitals of the charge=+2 A1 triplet. But if you change anything at all about this,
        # it drops by 2 orders of magnitude. So we probably want a different test here.
        self.assertAlmostEqual (e_roots[-1]-e_roots[-2], 5.39438435964712e-06, 10)

    def test_soc_2frag (self):
        e_ref = [-187.72602434, -187.53610235, -187.53600644, -187.50096683, -187.48344907]
        e_test, si_test = lasme2n2.lassi (opt=0, soc=True, break_symmetry=True)
        self.assertAlmostEqual (lib.fp (e_ref), lib.fp (e_test), 7)
        ## stationary test for >1 frag calc

    def test_soc_stdm12s (self):
        pass
        #stdm1s_test, stdm2s_test = make_stdm12s (lasme2n2, soc=True, opt=0)    
        ## stationary test for roots_make_stdm12s
  
    def test_soc_rdm12s (self):
        pass
        #rdm1s_test, rdm2s_test = roots_make_rdm12s (lasme2n2, lasme2n2.ci, si_ref, soc=True, opt=0)
        ## stationary test for roots_make_rdm12s

if __name__ == "__main__":
    print("Full Tests for SOC")
    unittest.main()
