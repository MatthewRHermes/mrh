import unittest
import numpy as np
from pyscf import gto, scf, mcscf
from mrh.my_pyscf.fci import csf_solver
from functools import reduce
from pyscf import lo, lib
from pyscf.pbc import gto, scf, dft, df
from DMET.my_pyscf.pdmet._pdfthelper import get_mc_for_dmet_pdft
from DMET.my_pyscf.pdmet import runpDMET 

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
        dmet_energy, core_energy, dmet_mf, trans_coeff = runpDMET(mf, lo_method='meta-lowdin', bath_tol=1e-10, atmlst=[0, 1])

        e_check = dmet_energy + core_energy

        # Sanity Check
        assert abs((mf.e_tot - e_check)) < 1e-7, "Something went wrong."

        # MC-PDFT: PBC-PDFT is in mrh only.
        from mrh.my_pyscf import mcpdft
        mc = mcpdft.CASSCF(mf, 'tPBE', 8, 10)
        mc.kernel()

        e_ref = mc.e_tot

        del mc

        from DMET.my_pyscf.pdmet._pdfthelper import get_mc_for_pdmet_pdft

        mc = mcscf.CASSCF(dmet_mf,8,10)
        mc.verbose = 0
        mc._scf.energy_nuc = lambda *args: core_energy 
        mc.fcisolver  = csf_solver(cell, smult=1)
        mc.kernel()

        newmc = get_mc_for_pdmet_pdft(mc, trans_coeff, mf)
        mypdft = mcpdft.CASCI(newmc, 'tPBE', mc.ncas, mc.nelecas)
        mypdft.compute_pdft_energy_(mo_coeff=newmc.mo_coeff, ci=mc.ci, dump_chk=False)

        e_check = mypdft.e_tot
      
        del cell, mf, dmet_energy, core_energy, dmet_mf, mc, newmc, mypdft
        self.assertAlmostEqual(e_ref, e_check, 6)


if __name__ == "__main__":
    # See the description of the tests at the top of the file.
    unittest.main()