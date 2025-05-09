import unittest
import numpy as np
from pyscf import gto, scf, mcscf
from pyscf.pbc import gto, scf
from pyscf.csf_fci import csf_solver
from mrh.my_pyscf.pdmet import runpDMET
from mrh.my_pyscf import mcpdft

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
        mf.verbose=0
        mf.run()
        dmet_mf, mypdmet = runpDMET(mf, lo_method='meta-lowdin', bath_tol=1e-10, atmlst=[0, 1])
        assert abs((mf.e_tot - dmet_mf.e_tot)) < 1e-7, "Something went wrong."

        mc = mcpdft.CASSCF(mf, 'tPBE', 8, 10).run()
        e_ref = mc.e_tot
        del mc

        mc = mcscf.CASSCF(dmet_mf,8,10)
        mc.fcisolver  = csf_solver(cell, smult=1)
        mc.kernel()
        
        mo_coeff = mypdmet.assemble_mo(mc.mo_coeff)
        mypdft = mcpdft.CASCI(mf, 'tPBE', mc.ncas, mc.nelecas)
        mypdft.compute_pdft_energy_(mo_coeff=mo_coeff, ci=mc.ci, dump_chk=False)
        e_check = mypdft.e_tot
        self.assertAlmostEqual(e_ref, e_check, 6)
        del cell, mf, dmet_mf, mc, mypdft, e_check, e_ref
        


if __name__ == "__main__":
    # See the description of the tests at the top of the file.
    unittest.main()
