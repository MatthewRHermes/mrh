import numpy as np
from pyscf import lib, mcscf
from pyscf.pbc import gto, scf, df
from mrh.my_pyscf.pdmet import runpDMET
from pyscf.csf_fci import csf_solver
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf.pdmet._pdfthelper import get_mc_for_pdmet_pdft

np.set_printoptions(precision=4)

# Define the cell
cell = gto.Cell(basis = 'gth-SZV',pseudo = 'gth-pade', a = np.eye(3) * 12, max_memory = 5000)
cell.atom = '''
N 0 0 0
N 0 0 1.1
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

dmet_mf, trans_coeff = runpDMET(mf, lo_method='meta-lowdin', bath_tol=1e-10, atmlst=[0, 1])

assert abs((mf.e_tot - dmet_mf.e_tot)) < 1e-7, "Something went wrong."

# MC-PDFT: PBC-PDFT is in mrh only.
# CASSCF Calculation
mc = mcscf.CASSCF(dmet_mf,8,10)
mc.fcisolver  = csf_solver(cell, smult=1)
mc.kernel()

newmc = get_mc_for_pdmet_pdft(mc, trans_coeff, mf)
mypdft = mcpdft.CASCI(newmc, 'tPBE', mc.ncas, mc.nelecas)
mypdft.compute_pdft_energy_(mo_coeff=newmc.mo_coeff, ci=mc.ci, dump_chk=False)

# SA-CASSCF Calculation
mc = mcscf.CASSCF(dmet_mf,8,10) 
mc.fcisolver  = csf_solver(cell, smult=1)
mc = mcscf.state_average_(mc, weights=[0.5, 0.5])
mc.kernel()

newmc = get_mc_for_pdmet_pdft(mc, trans_coeff, mf)
for i in range(len(mc.ci)):
    mypdft = mcpdft.CASCI(newmc, 'tPBE', mc.ncas, mc.nelecas)
    mypdft.compute_pdft_energy_(mo_coeff=newmc.mo_coeff, ci=mc.ci[i], dump_chk=False)

