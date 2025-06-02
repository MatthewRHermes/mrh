import os
import unittest
import numpy as np
from pyscf.pbc import gto, scf, df
from mrh.my_pyscf.pdmet import runpDMET

'''
***** RHF Embedding *****
1. Consider all the atoms in embedding space
2. Consider few atoms in embedding space
3. Consider few atoms in embedding space with density fitting
***** ROHF Embedding *****
1. Consider all the atoms in embedding space
2. Consider few atoms in embedding space
3. Consider few atoms in embedding space with density fitting
'''

def get_cell1(basis='gth-SZV', pseudo = 'gth-pade'):
    cell = gto.Cell(
        basis=basis, pseudo=pseudo, a=np.eye(3) * 12,
        max_memory=5000, verbose=0, output='/dev/null',
        atom='N 0 0 0; N 0 0 1.1; Ne 0 0 10'
    )
    cell.build()
    return cell

def get_cell2(basis='gth-SZV', pseudo = 'gth-pade'):
    cell = gto.Cell(
        atom='F 0 0 2; Ar 0 0 8;',
        basis=basis, pseudo=pseudo, a=np.eye(3) * 12,
        max_memory=5000, spin=1, verbose=0, output='/dev/null',
        )
    cell.build()
    return cell

def precomputed_gdf(cell):
    cderifile = 'pdmet_unittest.h5'
    gdf = df.GDF(cell)
    gdf._cderi_to_save=cderifile
    gdf.build()
    return cderifile

class KnownValues(unittest.TestCase):

    def test_rhf_variants(self):
        '''
        1. Considering all the atoms in embedding space.
        2. Only few atoms in embedding space.
        3. Density Fitting within the emb environment as well.
        '''
        for atmlst, density_fit in [([0,1,2], False), ([0], False), ([0], True)]:
            cell = get_cell1()
            mf = scf.RHF(cell, exxdiv=None).density_fit()
            mf.kernel()
            e_ref = mf.e_tot
            dmet_mf = runpDMET(mf, lo_method='lowdin', 
                               bath_tol=1e-10, atmlst=atmlst, 
                               density_fit=density_fit)[0]
            e_check = dmet_mf.e_tot
            del cell, mf, dmet_mf
            self.assertAlmostEqual(e_ref, e_check, 6)

    def test_vanilla_rohf(self):
        '''
        1. Considering all the atoms in embedding space.
        2. Only few atoms in embedding space.
        3. Density Fitting within the emb environment as well.
        '''
        for atmlst, density_fit in [([0,1,], False), ([0], False), ([0], True)]:
            cell = get_cell2()
            mf = scf.RHF(cell, exxdiv=None).density_fit()
            mf.kernel()
            e_ref = mf.e_tot
            dmet_mf = runpDMET(mf, lo_method='lowdin', 
                               bath_tol=1e-10, atmlst=atmlst, 
                               density_fit=density_fit)[0]
            e_check = dmet_mf.e_tot
            del cell, mf, dmet_mf
            
            self.assertAlmostEqual(e_ref, e_check, 6)

    def test_with_precomputed_gdf(self):
        # RHF
        for atmlst, density_fit in [([0,], True)]:
            cell = get_cell1()
            gdffile = precomputed_gdf(cell)
            mf = scf.RHF(cell, exxdiv=None).density_fit()
            mf.with_df._cderi = gdffile
            mf.kernel()
            e_ref = mf.e_tot
            dmet_mf = runpDMET(mf, lo_method='lowdin', 
                               bath_tol=1e-10, atmlst=atmlst, 
                               density_fit=density_fit)[0]
            e_check = dmet_mf.e_tot
            del cell, mf, dmet_mf
            egdffile = 'pdmet_unittest_df.h5'
            for f in [gdffile, egdffile]:
                if os.path.exists(f): os.remove(f)
            self.assertAlmostEqual(e_ref, e_check, 6)

        # ROHF
        for atmlst, density_fit in [([0], True)]:
            cell = get_cell2()
            gdffile = precomputed_gdf(cell)
            mf = scf.RHF(cell, exxdiv=None).density_fit()
            mf.with_df._cderi = gdffile
            mf.kernel()
            e_ref = mf.e_tot
            dmet_mf = runpDMET(mf, lo_method='lowdin', 
                               bath_tol=1e-10, atmlst=atmlst, 
                               density_fit=density_fit)[0]
            e_check = dmet_mf.e_tot
            del cell, mf, dmet_mf

            # Don't forget to remove emb gdf
            egdffile = 'pdmet_unittest_df.h5'
            for f in [gdffile, egdffile]:
                if os.path.exists(f): os.remove(f)
            self.assertAlmostEqual(e_ref, e_check, 6)
        
if __name__ == "__main__":
    # See the description of the tests at the top of the file.
    unittest.main()

