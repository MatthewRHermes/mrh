import unittest
import numpy as np
from pyscf import mcscf
from pyscf.pbc import gto, scf
from pyscf.csf_fci import csf_solver
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf.pdmet import runpDMET, addons

'''
***** PDMET-PDFT Embedding *****
1. Embedded PDFT = Full-System PDFT (For CASCI)
2. For Single Configuration=(1,1) Active Space Embedded PDFT =  CAS-PDFT  
'''

def get_cell(basis='gth-SZV', pseudo = 'gth-pade'):
    cell = gto.Cell(
        atom='F 0 0 2; Ar 0 0 8;',
        basis=basis, pseudo=pseudo, a=np.eye(3) * 12,
        max_memory=5000, spin=1, verbose=0, output='/dev/null',
        )
    cell.build()
    return cell

class KnownValues(unittest.TestCase):

    def test_casci_pdft(self):
        cell = get_cell()
        mf = scf.RHF(cell, exxdiv=None).density_fit()
        mf.kernel()

        orblst = addons.mo_comps(['F 2s', 'F 2p'], cell, 
                                 mf.mo_coeff, orth_method='meta-lowdin').argsort()
        # Generating the reference values
        mc = mcpdft.CASCI(mf, 'tPBE', 4, 7)
        mo = mc.sort_mo(orblst[-mc.ncas:], base=0)
        mc.fcisolver  = csf_solver(cell, smult=2)
        mc.kernel(mo)

        e_ref0 = mc._scf.e_tot
        e_ref1 = mc.e_cas
        e_ref2 = mc.e_tot

        del mc

        # Run the pDMET
        dmet_mf, mypdmet = runpDMET(mf, lo_method='lowdin', 
                            bath_tol=1e-10, atmlst=[0,], 
                            density_fit=True)
        
        # Active space orbitals
        mo_coeff = mypdmet.assemble_mo(dmet_mf.mo_coeff)
        orblst = addons.mo_comps(['F 2s', 'F 2p'], cell, mo_coeff, 
                                 orth_method='meta-lowdin').argsort()
        mypdft = mcpdft.CASCI(mf, 'tPBE', 4, 7)
        mo = mypdft.sort_mo(orblst[-mypdft.ncas:], base=0, mo_coeff=mo_coeff)
        mypdft.fcisolver  = csf_solver(cell, smult=2)
        mypdft.kernel(mo_coeff=mo)

        # Now compare:
        e_check0 = dmet_mf.e_tot
        e_check1 = mypdft.e_cas
        e_check2 = mypdft.e_tot

        del cell, mf, dmet_mf, mypdft, mypdmet

        self.assertAlmostEqual(e_ref0, e_check0, 6)
        self.assertAlmostEqual(e_ref1, e_check1, 6)
        self.assertAlmostEqual(e_ref2, e_check2, 6)

    def test_casscf_pdft(self):
        cell = get_cell()
        mf = scf.RHF(cell, exxdiv=None).density_fit()
        mf.kernel()

        orblst = addons.mo_comps(['F 2pz'], cell, 
                                 mf.mo_coeff, orth_method='meta-lowdin').argsort()
        # Generating the reference values
        mc = mcpdft.CASSCF(mf, 'tPBE', 1, 1)
        mo = mc.sort_mo(orblst[-mc.ncas:], base=0)
        mc.fcisolver  = csf_solver(cell, smult=2)
        mc.kernel(mo)
        del mc
        e_ref0 = mc._scf.e_tot
        e_ref1 = mc.e_cas
        e_ref2 = mc.e_tot

        # Run the pDMET
        dmet_mf, mypdmet = runpDMET(mf, lo_method='lowdin', 
                            bath_tol=1e-10, atmlst=[0,], 
                            density_fit=True)
        
        mc = mcscf.CASSCF(dmet_mf, 1, 1)
        mo = mc.sort_mo(orblst[-mc.ncas:], base=0)
        mc.fcisolver  = csf_solver(cell, smult=2)
        mc.kernel(mo)

        # Active space orbitals
        mo_coeff = mypdmet.assemble_mo(mc.mo_coeff)

        orblst = addons.mo_comps(['F 2pz'], cell, mo_coeff, 
                                 orth_method='meta-lowdin').argsort()
        mypdft = mcpdft.CASCI(mf, 'tPBE', 1, 1)
        mo = mypdft.sort_mo(orblst[-mypdft.ncas:], base=0, mo_coeff=mo_coeff)
        mypdft.fcisolver  = csf_solver(cell, smult=2)
        mypdft.kernel(mo_coeff=mo, ci0=mc.ci)

        # Now compare:
        e_check0 = dmet_mf.e_tot
        e_check1 = mypdft.e_cas
        e_check2 = mypdft.e_tot

        del cell, mf, dmet_mf, mc, mypdft, mypdmet

        self.assertAlmostEqual(e_ref0, e_check0, 6)
        self.assertAlmostEqual(e_ref1, e_check1, 6)
        self.assertAlmostEqual(e_ref2, e_check2, 6)

if __name__ == "__main__":
    # See the description of the tests at the top of the file.
    unittest.main()
