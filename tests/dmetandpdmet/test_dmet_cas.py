import unittest
from pyscf import gto, scf, mcscf
from mrh.my_pyscf.dmet import runDMET

'''
***** CAS-DMET Embedding *****
1. CASSCF==MF (1o, 2e) should be equals to the RHF Energy
2. CASSCF==MF (1o, 1e) should be equals to the ROHF Energy
3. CASSCF with all atoms in embedded space == CASSCF of full space.
4. SA-CASSCF?
5. CASCI tests
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

    def test_vanilla_casscf(self):
        mol = get_mole1()
        mf = scf.RHF(mol)
        mf.kernel()
        dmet_mf = runDMET(mf, lo_method='lowdin', bath_tol=1e-10, atmlst=[0,])[0]
        assert abs(mf.e_tot - dmet_mf.e_tot) < 1e-7, "Something went wrong."
        mc = mcscf.CASSCF(dmet_mf, 1, 2)
        mc.kernel()
        self.assertAlmostEqual(mf.e_tot, mc.e_tot, 6)
        del mol, mf, dmet_mf, mc
        
    def test_vanilla_casscf_openshell(self):
        mol = get_mole2()
        mf = scf.RHF(mol)
        mf.kernel()
        dmet_mf, mydmet = runDMET(mf, lo_method='lowdin', bath_tol=1e-10, atmlst=[0,])# Considering all the atoms in embedding space
        assert abs((mf.e_tot - dmet_mf.e_tot)) < 1e-7, "Something went wrong."
        mc = mcscf.CASSCF(dmet_mf, 1, 1)
        mc.kernel()
        self.assertAlmostEqual(mf.e_tot, mc.e_tot, 6)
        del mol, mf,  dmet_mf, mc, mydmet
        
    def test_casscf_to_non_emb(self):
        mol = get_mole2()
        mf = scf.RHF(mol)
        mf.kernel()
        mc = mcscf.CASSCF(mf, 2, 1)
        mc.kernel()
        e_ref = mc.e_tot
        del mc
        dmet_mf= runDMET(mf, lo_method='lowdin', bath_tol=1e-10, atmlst=[0,1,2])[0] # Considering all the atoms in embedding space
        assert abs((mf.e_tot - dmet_mf.e_tot)) < 1e-7, "Something went wrong."
        mc = mcscf.CASSCF(dmet_mf, 2, 1)
        mc.kernel()
        e_check = mc.e_tot
        self.assertAlmostEqual(e_ref, e_check, 6)
        del mol, mf, dmet_mf, mc

if __name__ == "__main__":
    # See the description of the tests at the top of the file.
    unittest.main()
