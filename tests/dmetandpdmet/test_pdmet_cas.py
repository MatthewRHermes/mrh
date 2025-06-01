import unittest
import numpy as np
from pyscf import mcscf
from pyscf.pbc import gto, scf
from pyscf.csf_fci import csf_solver
from mrh.my_pyscf.pdmet import runpDMET, addons

'''
***** PDMET-CAS Embedding *****
1. Testing CASCI: Embedded CASCI = Full-System CASCI (also testing more than one roots)
2. Testing CASSCF: For Single Configuration=(1,1) Active Space Embedded CAS =  CASSCF for the full systems
3. Testing SA-CASSCF: I don't have any other option than taking the entire space as embedding space then comparing. 
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

    def test_casci(self):
        cell = get_cell()
        mf = scf.RHF(cell, exxdiv=None).density_fit()
        mf.kernel()

        orblst = addons.mo_comps(['F 2s', 'F 2p'], cell, 
                                 mf.mo_coeff, orth_method='meta-lowdin').argsort()
        # Generating the reference values
        mc = mcscf.CASCI(mf, 4, 7)
        mo = mc.sort_mo(orblst[-mc.ncas:], base=0)
        mc.fcisolver.nroots = 2
        mc.fcisolver  = csf_solver(cell, smult=2)
        mc = mcscf.state_average_(mc, [0.5, 0.5]) # weight doesn't matter, still
        mc.kernel(mo)

        e_ref0 = mc._scf.e_tot
        e_ref1 = mc.e_states[0]
        e_ref2 = mc.e_states[1]

        del mc

        # Run the pDMET
        dmet_mf, mypdmet = runpDMET(mf, lo_method='lowdin', 
                            bath_tol=1e-10, atmlst=[0,], 
                            density_fit=True)
        
        # Active space orbitals
        mo_coeff = mypdmet.ao2eo @ dmet_mf.mo_coeff
        orblst = addons.mo_comps(['F 2s', 'F 2p'], cell, mo_coeff, 
                                 orth_method='meta-lowdin').argsort()
        
        mc = mcscf.CASCI(dmet_mf, 4, 7)
        mo = mc.sort_mo(orblst[-mc.ncas:], base=0)
        mc.fcisolver.nroots = 2
        mc.fcisolver  = csf_solver(cell, smult=2)
        mc = mcscf.state_average_(mc, [0.5, 0.5]) # weight doesn't matter, still
        mc.kernel(mo)

        e_check0 = dmet_mf.e_tot
        e_check1 = mc.e_states[0]
        e_check2 = mc.e_states[1]

        del cell, mf, dmet_mf, mc, mypdmet

        self.assertAlmostEqual(e_ref0, e_check0, 6)
        self.assertAlmostEqual(e_ref1, e_check1, 6)
        self.assertAlmostEqual(e_ref2, e_check2, 6)

    def test_casscf(self):
        cell = get_cell()
        mf = scf.RHF(cell, exxdiv=None).density_fit()
        mf.kernel()

        # Generating the reference values
        # orblst = addons.mo_comps(['F 2pz'], cell, 
        #                          mf.mo_coeff, orth_method='meta-lowdin').argsort()
        
        # mc = mcscf.CASSCF(mf, 1, 1)
        # mo = mc.sort_mo(orblst[-mc.ncas:], base=0)
        # mc.fcisolver  = csf_solver(cell, smult=2)
        # mc.kernel(mo)

        e_ref0 = mf.e_tot
        # e_ref1 = mc.e_tot 
        # They are the same energies
        e_ref1 = mf.e_tot
        
        # Run the pDMET
        dmet_mf, mypdmet = runpDMET(mf, lo_method='lowdin', 
                            bath_tol=1e-10, atmlst=[0,], 
                            density_fit=True)
        
        # Active space orbitals
        mo_coeff = mypdmet.ao2eo @ dmet_mf.mo_coeff
        orblst = addons.mo_comps(['F 2pz'], cell, 
                                 mo_coeff, orth_method='meta-lowdin').argsort()
        
        mc = mcscf.CASSCF(dmet_mf, 1, 1)
        mo = mc.sort_mo(orblst[-mc.ncas:], base=0)
        mc.fcisolver  = csf_solver(cell, smult=2)
        mc.kernel(mo)

        e_check0 = dmet_mf.e_tot
        e_check1 = mc.e_tot 

        del cell, mf, dmet_mf, mc, mypdmet

        self.assertAlmostEqual(e_ref0, e_check0, 6)
        self.assertAlmostEqual(e_ref1, e_check1, 6)

    def test_sa_casscf(self):
        cell = get_cell()
        mf = scf.RHF(cell, exxdiv=None).density_fit()
        mf.kernel()

        # Generating the reference values
        orblst = addons.mo_comps(['F 2s', 'F 2p'], cell, 
                                 mf.mo_coeff, orth_method='meta-lowdin').argsort()
        
        mc = mcscf.CASSCF(mf, 4, 7)
        mo = mc.sort_mo(orblst[-mc.ncas:], base=0)
        mc.fcisolver  = csf_solver(cell, smult=2)
        mc = mcscf.state_average_(mc, [0.5, 0.5])
        mc.kernel(mo)

        e_ref0 = mf.e_tot
        e_ref1 = mc.e_states[0]
        e_ref2 = mc.e_states[1]
        del mc
        # Run the pDMET
        dmet_mf, mypdmet = runpDMET(mf, lo_method='lowdin', 
                            bath_tol=1e-10, atmlst=[0,1], 
                            density_fit=True)
        
        # Active space orbitals
        mo_coeff = mypdmet.ao2eo @ dmet_mf.mo_coeff
        orblst = addons.mo_comps(['F 2pz'], cell, 
                                 mo_coeff, orth_method='meta-lowdin').argsort()
        
        mc = mcscf.CASSCF(dmet_mf, 4, 7)
        mo = mc.sort_mo(orblst[-mc.ncas:], base=0)
        mc.fcisolver  = csf_solver(cell, smult=2)
        mc = mcscf.state_average_(mc, [0.5, 0.5])
        mc.kernel(mo)

        e_check0 = dmet_mf.e_tot
        e_check1 = mc.e_states[0]
        e_check2 = mc.e_states[1] 

        del cell, mf, dmet_mf, mc, mypdmet

        self.assertAlmostEqual(e_ref0, e_check0, 6)
        self.assertAlmostEqual(e_ref1, e_check1, 6)
        self.assertAlmostEqual(e_ref2, e_check2, 6)

if __name__ == "__main__":
    # See the description of the tests at the top of the file.
    unittest.main()
