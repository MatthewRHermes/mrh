import numpy as np
from pyscf import gto, scf, mcscf, mcpdft
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.dmet import runDMET, getorbindex
from mrh.my_pyscf.dmet._pdfthelper import assemble_mo

'''
Note: In this example, I am only showing CAS-PDFT and MCPDFT based on the SA-CASSCF calculations. Basically
starting from the embedded CAS object, the 1RDMs and 2RDMs are projected to the entire space.
'''

np.set_printoptions(precision=4)

mol = gto.Mole(basis='6-31G', spin=1, charge=0, verbose=4, max_memory=10000)
mol.atom='''
P  -5.64983   3.02383   0.00000
H  -4.46871   3.02383   0.00000
H  -6.24038   2.19489   0.59928
Ne 0 0 10
'''
mol.build()

mf = scf.ROHF(mol).density_fit()
mf.kernel()

# Running DMET
dmet_mf, mydmet = runDMET(mf, lo_method='lowdin', bath_tol=1e-10, atmlst=[0, ])

# Sanity Check
assert abs((mf.e_tot - dmet_mf.e_tot)) < 1e-7, "Something went wrong."

# Active space guess
ao2eo = mydmet.ao2eo
ao2co = mydmet.ao2co

mo_coeff = ao2eo @ dmet_mf.mo_coeff
orblst = getorbindex(mol, mo_coeff, lo_method='meta-lowdin',
                    ao_label=['P 3s', 'P 3p', 'H 1s'], activespacesize=6, s=mf.get_ovlp())

# Running CASSCF
mc = mcscf.CASSCF(dmet_mf, 6, 7)
mc.max_cycle = 100
mo = mc.sort_mo(orblst)
mc.fcisolver  = csf_solver(mol, smult=2)
mc.kernel(mo)

# Assembling the full space mo_coeffs
mo_coeff = assemble_mo(mf, ao2eo, ao2co, mc.mo_coeff)

# Running CAS-PDFT
mypdft = mcpdft.CASCI(mf, 'tPBE', mc.ncas, mc.nelecas)
mypdft.kernel(mo_coeff=mo_coeff, ci0=mc.ci)

# MCPDFT based on SA-CAS Calculation
mc = mcscf.CASSCF(dmet_mf, 6, 7)
mo = mc.sort_mo(orblst)
mc.fcisolver  = csf_solver(mol, smult=2)
mc = mcscf.state_average_(mc, weights=[0.5, 0.5])
mc.kernel(mo)

# Assembling the full space mo_coeffs
mo_coeff = assemble_mo(mf, ao2eo, ao2co, mc.mo_coeff)

for i in range(len(mc.ci)):
    mypdft = mcpdft.CASCI(mf, 'tPBE', mc.ncas, mc.nelecas)
    mypdft.kernel(mo_coeff=mo_coeff, ci0=mc.ci[i])
