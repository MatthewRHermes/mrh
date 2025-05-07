import numpy as np
from pyscf import gto, scf, mcscf, mcpdft, lib
from mrh.my_pyscf.fci import csf_solver
from DMET.my_pyscf.dmet import runDMET, getorbindex

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

dmet_energy, core_energy, dmet_mf, trans_coeff = runDMET(mf, lo_method='lowdin', bath_tol=1e-10, atmlst=[0, ])

print("DMET:", dmet_energy)
print("Core Energy:", core_energy)
print("Total Energy", dmet_energy + core_energy)
print("Total Difference", mf.e_tot - (dmet_mf.e_tot + core_energy) )

# Sanity Check
assert abs((mf.e_tot - (dmet_mf.e_tot + core_energy))) < 1e-7, "Something went wrong."

# Active space guess
mo_coeff = trans_coeff['ao2eo'] @ dmet_mf.mo_coeff
orblst = getorbindex(mol, mo_coeff, lo_method='meta-lowdin',
                    ao_label=['P 3s', 'P 3p', 'H 1s'], activespacesize=6, s=mf.get_ovlp())

from DMET.my_pyscf.dmet._pdfthelper import get_mc_for_dmet_pdft

# MCPDFT based on SS-CAS Calculation
with lib.temporary_env(mf, exxdiv=None):
    mc = mcscf.CASSCF(dmet_mf, 6, 7)
    mc._scf.energy_nuc = lambda *args: core_energy
    mo = mc.sort_mo(orblst)
    mc.fcisolver  = csf_solver(mol, smult=2)
    mc.kernel(mo)
    newmc = get_mc_for_dmet_pdft(mc, trans_coeff, mf)

    # MC-PDFT
    mypdft = mcpdft.CASCI(newmc, 'tPBE', mc.ncas, mc.nelecas)
    mypdft.compute_pdft_energy_(mo_coeff=newmc.mo_coeff, ci=mc.ci, dump_chk=False)

# MCPDFT based on SA-CAS Calculation
with lib.temporary_env(mf, exxdiv=None):
    mc = mcscf.CASSCF(dmet_mf, 6, 7)
    mc._scf.energy_nuc = lambda *args: core_energy
    mo = mc.sort_mo(orblst)
    mc.fcisolver  = csf_solver(mol, smult=2)
    mc = mcscf.state_average_(mc, weights=[0.5, 0.5])
    mc.kernel(mo)

    # MC-PDFT for the more-than one state
    newmc = get_mc_for_dmet_pdft(mc, trans_coeff, mf)

    for i in range(len(mc.ci)):
        mypdft = mcpdft.CASCI(newmc, 'tPBE', mc.ncas, mc.nelecas)
        mypdft.compute_pdft_energy_(mo_coeff=newmc.mo_coeff, ci=mc.ci[i], dump_chk=False)
