import unittest
import numpy as np
from pyscf import gto, scf, mcscf
from pyscf.pbc import gto, scf
from pyscf.csf_fci import csf_solver
from mrh.my_pyscf.pdmet import runpDMET
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf.pdmet._pdfthelper import assemble_mo

'''
***** pDMET-PDFT Embedding *****
More tests has to be added.
'''

# # Integral generation
# gdf = df.GDF(cell)
# gdf._cderi_to_save = 'N2.h5'
# gdf.build()

# SCF: Note: use the density fitting object to build the SCF object

def get_cell():
    cell = gto.Cell(basis = 'gth-SZV',pseudo = 'gth-pade', a = np.eye(3) * 12, max_memory = 5000, verbose=0)
    cell.atom = '''
    N 0 0 0
    N 0 0 1.1
    '''
    cell.build()
    return cell


class KnownValues(unittest.TestCase):

    def test_pdft_closeshell(self):
        cell = get_cell()
        mf = scf.RHF(cell,exxdiv = None).density_fit()
        mf.verbose = 0
        mf.kernel()
        dmet_mf, trans_coeff = runpDMET(mf, lo_method='meta-lowdin', bath_tol=1e-10, atmlst=[0, 1])

        e_check = dmet_mf.e_tot

        # Sanity Check
        assert abs((mf.e_tot - e_check)) < 1e-7, "Something went wrong."

        mc = mcpdft.CASSCF(mf, 'tPBE', 8, 10)
        mc.kernel()

        e_ref = mc.e_tot

        del mc

        mc = mcscf.CASSCF(dmet_mf,8,10)
        mc.verbose = 0
        mc.fcisolver  = csf_solver(cell, smult=1)
        mc.kernel()

        mo_coeff = assemble_mo(mf, trans_coeff, mc.mo_coeff)
        mypdft = mcpdft.CASCI(mf, 'tPBE', mc.ncas, mc.nelecas)
        mypdft.compute_pdft_energy_(mo_coeff=mo_coeff, ci=mc.ci, dump_chk=False)


        e_check = mypdft.e_tot
      
        del cell, mf, dmet_mf, mc, mypdft
        self.assertAlmostEqual(e_ref, e_check, 6)


if __name__ == "__main__":
    # See the description of the tests at the top of the file.
    unittest.main()
