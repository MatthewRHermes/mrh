import unittest
import numpy as np
from pyscf import gto, scf, mcscf, mcpdft
from DMET.my_pyscf.dmet._pdfthelper import get_mc_for_dmet_pdft
from DMET.my_pyscf.dmet import runDMET

'''
***** DMET-PDFT Embedding *****
Few more tests has to be added.
1. Hybrid functional
2. SA-CAS Tests
'''

def get_mole1():
    mol = gto.Mole(basis='6-31G', spin=0, charge=0, verbose=0)
    mol.atom = '''
    S  -5.64983   3.02383   0.00000
    H  -4.46871   3.02383   0.00000
    H  -6.24038   2.19489   0.59928
    '''
    mol.build()
    return mol

def get_mole2():
    mol = gto.Mole(basis='6-31G', spin=1, charge=0, verbose=0)
    mol.atom = '''
    P  -5.64983   3.02383   0.00000
    H  -4.46871   3.02383   0.00000
    H  -6.24038   2.19489   0.59928
    '''
    mol.build()
    return mol

class KnownValues(unittest.TestCase):

    def test_pdft_closeshell(self):
        mol = get_mole1()

        mf = scf.RHF(mol)
        mf.kernel()
        
        mc = mcpdft.CASSCF(mf, 'tPBE', 1, 2)
        mc.kernel()

        e_ref = mc.e_tot
        
        dmet_energy, core_energy, dmet_mf, trans_coeff = runDMET(mf, lo_method='lowdin', bath_tol=1e-10, atmlst=[0,1,2])
        e_check = dmet_energy + core_energy

        # Sanity Check
        assert abs((mf.e_tot - e_check)) < 1e-7, "Something went wrong."

        # CASSCF Calculation
        mc = mcscf.CASSCF(dmet_mf, 1, 2)
        mc._scf.energy_nuc = lambda *args: core_energy
        mc.kernel()
        
        newmc = get_mc_for_dmet_pdft(mc, trans_coeff, mf)
        
        mypdft = mcpdft.CASCI(newmc, 'tPBE', mc.ncas, mc.nelecas)
        mypdft.compute_pdft_energy_(mo_coeff=newmc.mo_coeff, ci=mc.ci, dump_chk=False)

        e_check = mypdft.e_tot
      
        del mol, mf, dmet_energy, core_energy, dmet_mf, mc, newmc, mypdft
        self.assertAlmostEqual(e_ref, e_check, 6)
    
    def test_pdft_openshell(self):
        mol = get_mole2()

        mf = scf.RHF(mol)
        mf.kernel()

        # CASPDFT 'tPBE' with same AS as below: Instead of re-running I am 
        # saving this value due to circular dependencies.

        e_ref = -342.1818806123503 

        dmet_energy, core_energy, dmet_mf, trans_coeff = runDMET(mf, lo_method='lowdin', bath_tol=1e-10, atmlst=[0,1,2])
        e_check = dmet_energy + core_energy

        # Sanity Check
        assert abs((mf.e_tot - e_check)) < 1e-7, "Something went wrong."

        # CASSCF Calculation
        mc = mcscf.CASSCF(dmet_mf, 1, 1)
        mc._scf.energy_nuc = lambda *args: core_energy
        mc.kernel()
        
        newmc = get_mc_for_dmet_pdft(mc, trans_coeff, mf)
        
        mypdft = mcpdft.CASCI(newmc, 'tPBE', mc.ncas, mc.nelecas)
        mypdft.compute_pdft_energy_(mo_coeff=newmc.mo_coeff, ci=mc.ci, dump_chk=False)

        e_check = mypdft.e_tot
      
        del mol, mf, dmet_energy, core_energy, dmet_mf, mc, newmc, mypdft
        self.assertAlmostEqual(e_ref, e_check, 6)

    def test_pdft_openshell_2(self):
        mol = get_mole2()

        mf = scf.RHF(mol)
        mf.kernel()

        # CASPDFT 'tPBE' with same AS as below: Instead of re-running I am 
        # saving this value due to circular dependencies.
        e_ref = -342.1818806123501

        dmet_energy, core_energy, dmet_mf, trans_coeff = runDMET(mf, lo_method='lowdin', bath_tol=1e-10, atmlst=[0,1,2])
        e_check = dmet_energy + core_energy

        # Sanity Check
        assert abs((mf.e_tot - e_check)) < 1e-7, "Something went wrong."

        # CASSCF Calculation
        mc = mcscf.CASSCF(dmet_mf, 2, 1)
        mc._scf.energy_nuc = lambda *args: core_energy
        mc.kernel()
        
        newmc = get_mc_for_dmet_pdft(mc, trans_coeff, mf)
        
        mypdft = mcpdft.CASCI(newmc, 'tPBE', mc.ncas, mc.nelecas)
        mypdft.compute_pdft_energy_(mo_coeff=newmc.mo_coeff, ci=mc.ci, dump_chk=False)

        e_check = mypdft.e_tot
      
        del mol, mf, dmet_energy, core_energy, dmet_mf, mc, newmc, mypdft
        self.assertAlmostEqual(e_ref, e_check, 6)

if __name__ == "__main__":
    # See the description of the tests at the top of the file.
    unittest.main()
