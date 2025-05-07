from pyscf import lib
from pyscf.pbc import gto, scf, df
from DMET.my_pyscf.pdmet import runpDMET 
import numpy as np
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

dmet_energy, core_energy, dmet_mf, trans_coeff = runpDMET(mf, lo_method='meta-lowdin', bath_tol=1e-10, atmlst=[0, 1])

print("DMET:", dmet_energy)
print("Core Energy:", core_energy)
print("Total Energy", dmet_energy + core_energy)
print("Total Difference", mf.e_tot - (dmet_mf.e_tot + core_energy) )
assert abs((mf.e_tot - (dmet_mf.e_tot + core_energy))) < 1e-7, "Something went wrong."


from pyscf import mcscf, mrpt
from mrh.my_pyscf.fci import csf_solver

# CASSCF Calculation
with lib.temporary_env(dmet_mf, exxdiv=None):
    mc = mcscf.CASSCF(dmet_mf,8,10)
    mc._scf.energy_nuc = lambda *args: core_energy 
    mc.fcisolver  = csf_solver(cell, smult=1)
    mc.kernel()
    
    # In case of natorb=True, CAS will try to expand the active space in AO basis for verbose>3, and AO basis for the 
    # embedded space is not defined. Therefore, for the NEVPT2 and natorb=True, we need to set verbose <= 3.
    mc.natorb = True
    mc.verbose = 3
    mc.kernel()

    e_corr = mrpt.NEVPT(mc).kernel()
    e_tot = mc.e_tot + e_corr
    print('NEVPT2 energy: ', e_tot)

 
# SA-CASSCF Calculation
with lib.temporary_env(dmet_mf, exxdiv=None):
    mc = mcscf.CASSCF(dmet_mf,8,10)
    mc._scf.energy_nuc = lambda *args: core_energy 
    mc.fcisolver  = csf_solver(cell, smult=1)
    mc = mcscf.state_average_(mc, weights=[0.5, 0.5])
    mc.kernel()

    newmc = mcscf.CASCI(dmet_mf, 8, 10)
    newmc.verbose = 3
    newmc.natorb = True
    newmc.fcisolver.nroots = len(mc.ci)
    newmc._scf.energy_nuc = lambda *args: core_energy
    newmc.kernel(mc.mo_coeff)

    for i in range(len(newmc.ci)):
        e_corr = mrpt.NEVPT(newmc,root=i).kernel()
        e_tot = newmc.e_tot[i] + e_corr
        print(f'NEVPT2 energy for state {i}: ', e_tot)
        


