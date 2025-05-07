import numpy as np
from pyscf import gto, scf, mcscf, mrpt
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.dmet import runDMET, getorbindex

np.set_printoptions(precision=4)

'''
NEVPT2 calculation in embedded DMET environment.
It's not that straightforward due the way the NEVPT2 code is written.
1. For make NEVPT2 to run, with current settings, we need to use density fitting=False for the embedded space.
2. The verbose of the mc object has to be lower than or equal to 3, otherwise pyscf
    will try to expand the active space in AO basis for verbose>3, and AO basis for the embedded space is not defined.
'''

'''
This also means, for using mc.natorb = True, make verbosity <= 3.
'''

mol = gto.Mole(basis='6-31G', spin=1, charge=0, verbose=4, max_memory=10000)
mol.atom='''
P  -5.64983   3.02383   0.00000
H  -4.46871   3.02383   0.00000
H  -6.24038   2.19489   0.59928
He 0 0 10
'''
mol.build()

mf = scf.ROHF(mol).density_fit()
mf.kernel()

# Set the density fitting to False, by default it is True.
dmet_mf, trans_coeff = runDMET(mf, lo_method='lowdin', bath_tol=1e-10, atmlst=[0, ], density_fit=False)

# Sanity Check
assert abs((mf.e_tot - dmet_mf.e_tot)) < 1e-7, "Something went wrong."

# Active space guess
mo_coeff = trans_coeff['ao2eo'] @ dmet_mf.mo_coeff
orblst = getorbindex(mol, mo_coeff, lo_method='meta-lowdin',
                    ao_label=['P 3s', 'P 3p', 'H 1s'], activespacesize=6, s=mf.get_ovlp())

# CASSCF Calculation
mc = mcscf.CASSCF(dmet_mf, 6, 7)
mo = mc.sort_mo(orblst)
mc.fcisolver  = csf_solver(mol, smult=2)
mc.kernel(mo)

# In case of natorb=True, CAS will try to expand the active space in AO basis for verbose>3, and AO basis for the 
# embedded space is not defined. Therefore, for the NEVPT2 and natorb=True, we need to set verbose <= 3.
mc.natorb = True
mc.verbose = 3
mc.kernel()

e_corr = mrpt.NEVPT(mc).kernel()
e_tot = mc.e_tot + e_corr
print('NEVPT2 energy: ', e_tot)

# SA-CASSCF Calculation
mc = mcscf.CASSCF(dmet_mf, 6, 7)
mo = mc.sort_mo(orblst)
mc.fcisolver  = csf_solver(mol, smult=2)
mc = mcscf.state_average_(mc, weights=[0.5, 0.5])
mc.kernel(mo)

newmc = mcscf.CASCI(dmet_mf, 6, 7)
newmc.verbose = 3
newmc.natorb = True
newmc.fcisolver.nroots = len(mc.ci)
newmc.kernel(mc.mo_coeff)

for i in range(len(newmc.ci)):
    e_corr = mrpt.NEVPT(newmc,root=i).kernel()
    e_tot = newmc.e_tot[i] + e_corr
    print(f'NEVPT2 energy for state {i}: ', e_tot)
        
