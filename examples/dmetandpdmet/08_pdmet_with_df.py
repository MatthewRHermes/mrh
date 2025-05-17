
import numpy as np
from pyscf import mcscf
from pyscf.pbc import gto, scf, df
from pyscf.csf_fci import csf_solver
from mrh.my_pyscf.pdmet import runpDMET 

np.set_printoptions(precision=4)

# Define the cell
cell = gto.Cell(basis = 'gth-SZV',pseudo = 'gth-pade', a = np.eye(3) * 20, max_memory = 5000)
cell.atom = '''
N 0 0 0
N 0 0 1.1
Ne 0 0 18
'''
cell.verbose = 4
cell.build()

# Integral generation
gdf = df.GDF(cell)
gdf._cderi_to_save = 'N2.h5'
gdf.build()

# SCF: Note: use the density fitting object to build the SCF object
mf = scf.RHF(cell).density_fit()
mf.exxdiv = None
mf.with_df._cderi = 'N2.h5'
mf.kernel()

# In this case, we will be using density_fitting in the embedded part of the molecule as well.
dmet_mf, mypdmet = runpDMET(mf, lo_method='meta-lowdin', bath_tol=1e-10, atmlst=[0, 1], density_fit=True)
assert abs((mf.e_tot - dmet_mf.e_tot)) < 1e-7, "Something went wrong."

# CASCI Calculation
mc = mcscf.CASSCF(dmet_mf,8,10)
mc.fcisolver  = csf_solver(cell, smult=1)
mc.kernel()

# CASSCF Calculation
mc = mcscf.CASSCF(dmet_mf,8,10)
mc.fcisolver  = csf_solver(cell, smult=1)
mc.kernel()

