import numpy as np
from scipy import linalg
from pyscf import gto, scf, lib, mcscf
from pyscf.data.nist import HARTREE2EV
from pyscf.fci.direct_spin1 import _unpack_nelec
from mrh.my_pyscf import mcpdft
import unittest

Natom = scf.RHF (gto.M (atom = 'N 0 0 0', basis='cc-pvtz', spin=3, symmetry='Dooh', output='/dev/null'))
Beatom = scf.RHF (gto.M (atom = 'Be 0 0 0', basis='cc-pvtz', spin=2, symmetry=False, output='/dev/null'))
Natom_hs = [Natom, (4,1), None]
Natom_ls = [Natom, (3,2), None]
Beatom_ls = [Beatom, (1,1), None]
Beatom_hs = [Beatom, (2,0), None]

def check_calc (calc):
    if not calc[0].converged: calc[0].kernel ()
    if calc[-1] is None:
        mf = calc[0]
        mol = mf.mol
        nelecas = calc[1]
        s = (nelecas[1]-nelecas[0])*0.5
        ss = s*(s+1)
        mc = mcscf.CASSCF (mf, 4, nelecas).set (conv_tol=1e-10)
        mc.fix_spin_(ss=ss)
        calc[-1] = mc
    if not calc[-1].converged:
        calc[-1].kernel ()
    return calc[-1]

def get_gap (gs, es, fnal):
    gs = check_calc (gs)
    es = check_calc (es)
    e0 = mcpdft.CASSCF (gs._scf, fnal, gs.ncas, gs.nelecas, grids_level=9).set (
        fcisolver = gs.fcisolver, conv_tol=1e-10).kernel (
        gs.mo_coeff, gs.ci)[0]
    e1 = mcpdft.CASSCF (es._scf, fnal, es.ncas, es.nelecas, grids_level=9).set (
        fcisolver = es.fcisolver, conv_tol=1e-10).kernel (
        es.mo_coeff, es.ci)[0]
    return (e1-e0)*HARTREE2EV

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






