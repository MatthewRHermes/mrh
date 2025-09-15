import numpy as np
from pyscf import gto, scf, mcscf, lib, siso
from pyscf.csf_fci import csf_solver
from mrh.my_pyscf.dmet import runDMET
from mrh.my_pyscf.gto import get_ano_rcc_basis

'''
SOC interactions with-in DMET Framework at QDPT level using BP or DK Hamiltonian:
One can use this with following methods CAS, MC-PDFT, NEVPT2, and L-PDFT.
'''

'''
Currently, the SOC is hosted on my local fork of pyscf-forge:
https://github.com/JangidBhavnesh/pyscf-forge/tree/qdptsoclpdft
In future, I will try to push this to main pyscf-forge repo.
'''

np.set_printoptions(precision=4)

mol = gto.Mole(spin=1, charge=0, verbose=4, max_memory=100000)
mol.atom='''
Ne 0 0 -10
B 0.0 0.0 0.0
Ne 0 0 10
'''
basis = get_ano_rcc_basis (mol, 'VDZP')
mol.basis = basis
mol.build()

mf = scf.ROHF(mol).sfx2c1e().density_fit()
mf.kernel()

dmet_mf, mydmet = runDMET(mf, lo_method='lowdin', bath_tol=1e-6, atmlst=[1, ])

# Sanity Check
assert abs((mf.e_tot - dmet_mf.e_tot)) < 1e-7, "Something went wrong."

# Active space guess
ao2eo = mydmet.ao2eo
ao2co = mydmet.ao2co
mo_coeff = ao2eo @ dmet_mf.mo_coeff

# Based on the AO Character
from pyscf.tools import mo_mapping
orblst = mo_mapping.mo_comps(['B 2s', 'B 2p'], mol, mo_coeff, orth_method='meta-lowdin')
orblst = orblst.argsort()

# CASSCF Calculation
mc = mcscf.CASSCF(dmet_mf, 4, 3)
mc = siso.sacasscf_solver(mc, [(3, 2),])
mo = mc.sort_mo(orblst[-mc.ncas:], base=0)
mc.kernel(mo)

# Assembling the full space mo_coeffs
mo_coeff = mydmet.assemble_mo(mc.mo_coeff)

# Once you have the entire molecule mo_coeff, you can compute the SOC interactions
# For CAS, PDFT, PT2 or L-PDFT. Here, I am only showing CAS, but should be straightforward
# to use this with other methods by looking at examples pyscf-forge/examples/siso/*.py

mc = mcscf.CASCI(mf, mc.ncas, mc.nelecas)
mc = siso.sacasscf_solver(mc, [(3, 2),])
mc.kernel(mo_coeff=mo_coeff)

mysiso = siso.SISO(mc,  [(3, 2),], ham='DK', amf=True)
mysiso.kernel()


