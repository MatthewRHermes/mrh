import copy
import unittest
import numpy as np
from scipy import linalg
from pyscf import lib, gto, scf
from pyscf.lo import orth
from mrh.my_pyscf.mcscf import lasscf_o0
from mrh.my_pyscf.lassi import LASSIrq, LASSIS, lassi

# The purpose of this test is to make sure the LASSI kernel doesn't try to build an array with
# nfrags dimensions at any point in the calculation.

def setUpModule():
    global las
    atom = '\n'.join (['H {} 0 0'.format (i) for i in range (33)])
    mol = gto.M (atom=atom, basis='6-31g', symmetry=False, spin=33, verbose=0, output='/dev/null')
    mf = scf.RHF (mol)
    nlas = [2,]*33
    spin_sub = [2,]*33
    nelelas = [[1,0],]*33
    las = lasscf_o0.LASSCF (mf, nlas, nelelas, spin_sub=spin_sub)
    las.max_cycle_macro = 1
    las.mo_coeff = orth.orth_ao (mol, 'meta_lowdin', s=mf.get_ovlp ())
    las.lasci ()

def tearDownModule():
    global las
    del las

class KnownValues(unittest.TestCase):

    def test_lassi01 (self):
        lsi = LASSIrq (las, 0, 1).run ()
        lsid = LASSIrq (las, 0, 1).run (davidson_only=True)
        rdm1s, rdm2s = lassi.root_make_rdm12s(lsi, lsi.ci, lsi.si, state=0)
        rdm1s, rdm2s = lassi.root_make_rdm12s(lsid, lsid.ci, lsid.si, state=0)

if __name__ == "__main__":
    print("Full Tests for LASSI 33-fragment calculation")
    unittest.main()


