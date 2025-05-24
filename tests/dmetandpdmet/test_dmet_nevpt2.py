import unittest
from pyscf import gto, scf, mcscf, mrpt
from mrh.my_pyscf.dmet import runDMET

'''
***** NEVPT2-DMET Embedding *****
1. OpenShell: based on CASCI
2. CloseShell: based on CASCI
Basically, CASCI is the same, hence the PT2 should be same, no! it depends the orbital space. Hence
I will consider the entire space in the embedded part.
'''

# Note NEVPT2 in the embedding space can't be solved with Density Fitting in current implementation.

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
            
    def test_pt2_forcloseshell(self):
        mol = get_mole1()
        mf = scf.RHF(mol)
        mf.kernel()

        mc = mcscf.CASSCF(mf, 1, 2)
        mc.kernel()
        e_corr = mrpt.NEVPT(mc,root=1).kernel()
        e_ref1 = mc.e_tot + e_corr
        del mc

        dmet_mf= runDMET(mf, lo_method='lowdin', bath_tol=1e-10, atmlst=[0,1,2])[0] # Considering all the atoms in embedding space
        assert abs((mf.e_tot - dmet_mf.e_tot)) < 1e-7, "Something went wrong."

        mc = mcscf.CASSCF(dmet_mf, 1, 2)
        mc.kernel()
        e_corr = mrpt.NEVPT(mc,root=1).kernel()
        e_check1 = mc.e_tot + e_corr

        self.assertAlmostEqual(e_ref1, e_check1, 6)
        del mol, mf, dmet_mf, mc

    def test_pt2_foropenshell(self):
        mol = get_mole2()
        mf = scf.RHF(mol)
        mf.kernel()

        mc = mcscf.CASSCF(mf, 2, 1)
        mc.kernel()
        e_corr = mrpt.NEVPT(mc,root=1).kernel()
        e_ref1 = mc.e_tot + e_corr
        del mc

        dmet_mf= runDMET(mf, lo_method='lowdin', bath_tol=1e-10, atmlst=[0,1,2])[0] # Considering all the atoms in embedding space
        assert abs((mf.e_tot - dmet_mf.e_tot)) < 1e-7, "Something went wrong."

        mc = mcscf.CASSCF(dmet_mf, 2, 1)
        mc.kernel()
        e_corr = mrpt.NEVPT(mc,root=1).kernel()
        e_check1 = mc.e_tot + e_corr

        self.assertAlmostEqual(e_ref1, e_check1, 6)
        del mol, mf, dmet_mf, mc

if __name__ == "__main__":
    # See the description of the tests at the top of the file.
    unittest.main()
