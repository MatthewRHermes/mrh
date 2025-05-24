import unittest
import numpy as np
from pyscf import mcscf, mrpt
from pyscf.pbc import gto, scf
from pyscf.csf_fci import csf_solver
from mrh.my_pyscf.pdmet import runpDMET, addons

'''
***** PDMET-NEVPT2 Embedding *****
Only one test:
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
        mc = mcscf.CASSCF(mf, 1, 1)
        mo = mc.sort_mo(orblst[-mc.ncas:], base=0)
        mc.fcisolver  = csf_solver(cell, smult=2)
        mc.kernel(mo)

        e_ref0 = mc._scf.e_tot
        e_ref1 = mc.e_tot

        e_corr = mrpt.NEVPT(mc,root=1).kernel()
        e_ref2 = mc.e_tot + e_corr
        del mc

        # Run the pDMET
        dmet_mf, mypdmet = runpDMET(mf, lo_method='lowdin', 
                            bath_tol=1e-10, atmlst=[0,1], 
                            density_fit=True)
        
        # Active space orbitals
        mo_coeff = mypdmet.ao2eo @ dmet_mf.mo_coeff
        orblst = addons.mo_comps(['F 2s', 'F 2p'], cell, mo_coeff, 
                                 orth_method='meta-lowdin').argsort()
        
        mc = mcscf.CASSCF(dmet_mf, 1, 1)
        mo = mc.sort_mo(orblst[-mc.ncas:], base=0)
        mc.fcisolver  = csf_solver(cell, smult=2)
        mc.kernel(mo)

        e_corr = mrpt.NEVPT(mc,root=1).kernel()
        e_check0 = dmet_mf.e_tot
        e_check1 = mc.e_tot
        e_check2 = mc.e_tot + e_corr

        del cell, mf, dmet_mf, mc, mypdmet

        self.assertAlmostEqual(e_ref0, e_check0, 6)
        self.assertAlmostEqual(e_ref1, e_check1, 6)
        self.assertAlmostEqual(e_ref2, e_check2, 6)

if __name__ == "__main__":
    # See the description of the tests at the top of the file.
    unittest.main()
