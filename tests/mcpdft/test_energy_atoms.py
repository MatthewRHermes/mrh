import numpy as np
from scipy import linalg
from pyscf import gto, scf, lib, mcscf
from pyscf.fci.direct_spin1 import _unpack_nelec
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf.fci import csf_solver
import unittest

Natom = scf.RHF (gto.M (atom = 'N 0 0 0', basis='cc-pvtz', spin=3, symmetry='Dooh', output='/dev/null')).run ()
Beatom = scf.RHF (gto.M (atom = 'Be 0 0 0', basis='cc-pvtz', spin=2, symmetry=False, output='/dev/null')).run ()
Natom_hs = mcscf.CASSCF (Natom, 4, (4,1)).set (fcisolver=csf_solver (Natom.mol, smult=4), conv_tol=1e-10).run ()
Natom_ls = mcscf.CASSCF (Natom, 4, (3,2)).set (fcisolver=csf_solver (Natom.mol, smult=2), conv_tol=1e-10).run ()
Beatom_ls = mcscf.CASSCF (Beatom, 4, (1,1)).set (fcisolver=csf_solver (Beatom.mol, smult=1), conv_tol=1e-10).run ()
Beatom_hs = mcscf.CASSCF (Beatom, 4, (2,0)).set (fcisolver=csf_solver (Beatom.mol, smult=3), conv_tol=1e-10).run ()

def get_gap (gs, es, fnal):
    e0 = mcpdft.CASSCF (gs._scf, fnal, gs.ncas, gs.nelecas, grids_level=9).set (
        fcisolver = gs.fcisolver, conv_tol=1e-10).kernel (
        gs.mo_coeff, gs.ci)[0]
    e1 = mcpdft.CASSCF (es._scf, fnal, es.ncas, es.nelecas, grids_level=9).set (
        fcisolver = es.fcisolver, conv_tol=1e-10).kernel (
        es.mo_coeff, es.ci)[0]
    return (e1-e0)*27.2114

def tearDownModule():
    global Natom, Natom_hs, Natom_ls, Beatom, Beatom_ls, Beatom_hs
    Natom.mol.stdout.close ()
    Beatom.mol.stdout.close ()
    del Natom, Natom_hs, Natom_ls, Beatom, Beatom_ls, Beatom_hs

class KnownValues(unittest.TestCase):

    def test_tpbe_be (self):
        self.assertAlmostEqual (get_gap (Beatom_ls, Beatom_hs, 'tpbe'), 2.5670836958424603, 5) # Agrees w/ Manni 2014
        
    def test_tpbe_n (self):
        self.assertAlmostEqual (get_gap (Natom_hs, Natom_ls, 'tpbe'), 2.057962022799115, 5) # Agrees w/ Manni 2014

    def test_tblyp_be (self):
        self.assertAlmostEqual (get_gap (Beatom_ls, Beatom_hs, 'tblyp'), 2.5679364213750278, 5) # Agrees w/ Manni 2014
        
    def test_tblyp_n (self):
        self.assertAlmostEqual (get_gap (Natom_hs, Natom_ls, 'tblyp'), 1.9849000115274549, 5) # Agrees w/ Manni 2014

    def test_ftpbe_be (self):
        self.assertAlmostEqual (get_gap (Beatom_ls, Beatom_hs, 'ftpbe'), 2.5978084400532353, 5)
        
    def test_ftpbe_n (self):
        self.assertAlmostEqual (get_gap (Natom_hs, Natom_ls, 'ftpbe'), 1.4063841258726395, 5)

    def test_ftblyp_be (self):
        self.assertAlmostEqual (get_gap (Beatom_ls, Beatom_hs, 'ftblyp'), 2.584767118218244, 5)
        
    def test_ftblyp_n (self):
        self.assertAlmostEqual (get_gap (Natom_hs, Natom_ls, 'ftblyp'), 1.223835224013092, 5)

if __name__ == "__main__":
    print("Full Tests for MC-PDFT energies of N and Be atom spin states")
    unittest.main()






