import numpy as np
import os
from functools import reduce
from pyscf import fci
from pyscf import lib
from pyscf.pbc import scf, df, ao2mo
from pyscf.pbc import gto as pgto
from mrh.my_pyscf.pbc.mcscf import avas
from mrh.my_pyscf.pbc.fci import direct_com_real
from mrh.my_pyscf.pbc.mcscf.k2R import actmo_k2R, actmo_R2k, get_mo_coeff_k2R

'''
Aim: Generate the electronic Hamiltonian, and use that to solve
the FCI problem. For the Gamma point you can directly compare it with the mcscf.CASCI, check if both the energies are matching.
'''

def get_xyz(nU=1, d= 2.47):
    '''
    Generate atomic coordinates for a system with nU unit cells.
    args:
        nU: Number of unit cells
        d:  lattice vector of UC
    '''
    coords = [
    ("C", -0.5892731038,  0.3262391909,  0.0000000000),
    ("H", -0.5866101958,  1.4126530287,  0.0000000000),
    ("C",  0.5916281105, -0.3261693897,  0.0000000000),
    ("H",  0.5889652025, -1.4125832275,  0.0000000000)]

    translated_coords = []
    for t in range(nU):
        shift = t * d
        translated_coords.extend([(elem, x + shift, y, z)
            for elem, x, y, z in coords])
    return translated_coords

def get_gdf(filename, kpts=None, restart=True):
    """
    Calculate the 2e Integrals using the Gaussian Density Fitting.
    """
    if not os.path.exists(filename) or restart:
        gdf = df.GDF(cell, kpts=kpts)
        gdf._cderi_to_save = filename
        gdf.build()
    return filename

# Build the Cell object
nU = 1
d = 2.47
basis = '6-31G'
pseudo = None
maxMem = 120000

cell = pgto.Cell(atom = get_xyz(nU, d),
                 a = np.diag([d*nU, 17.5, 17.5]),
                 basis = basis,
                 pseudo = pseudo,
                 precision = 1e-10,
                 verbose = 3, #lib.logger.INFO,
                 max_memory = maxMem,
                 ke_cutoff = 40,
)
cell.build()

nk = [4, 1, 1]
kpts = cell.make_kpts(nk)
nC = nU * nk[0]

kmf = scf.KRHF(cell, kpts=kpts).density_fit(auxbasis='def2-svp-jkfit')
kmf.max_cycle=1000
kmf.chkfile = f'PAchain.{nC}.chk'
kmf.with_df._cderi = get_gdf(f'PAchain.{nC}.gdf', kpts, restart=False)
kmf.exxdiv = None
kmf.init_guess = 'chk'
kmf.conv_tol = 1e-10
meanfieldenergy = kmf.kernel()

# Use the right AVAS, for the active space selection.
ncas, nelecas, mo_coeff = avas.kernel(kmf, ['C 2pz'], minao=cell.basis, threshold=0.01, canonicalize=True)


# Steps
# 1. Compute the core energy
# 2. Compute the one-electron integrals in AO basis
# 3. Compute the two-electron integrals in AO basis
# 4. Transform the integrals to MO basis
# 5. Solve the FCI problem

from pyscf.tools import molden

# Generate the CAS Hamiltonian
def _basis_transformation(mat, mo_coeff):
    return reduce(np.dot, (mo_coeff.conj().T, mat, mo_coeff))

def active_space_eri(scell, mo_coeff_R, filename):
    '''
    Get the active space eri in real-space
    '''
    mf = scf.RHF(scell).density_fit('def2-svp-jkfit')
    mf.exxdiv = None
    mf.with_df._cderi = filename
    if not os.path.exists(filename):
        mf.with_df._cderi_to_save = filename
        mf.with_df.build()
    eri = mf.with_df.ao2mo(mo_coeff_R,)
    return eri

def get_coredm(mo_c, nkpts=1):
    '''
    Basically the cdm is 2* (C @ C.conj().T)
    '''
    core_dm_k = np.asarray([2.0 * (mo_c[k] @ mo_c[k].conj().T) for k in range(nkpts)], dtype=mo_c[0].dtype)
    return core_dm_k

def get_fci_ham(kmf, ncore, ncas, mo_coeff, eri_file):
    '''
    Get the FCI Hamiltonian in the CAS space.
    First compute the one-e Ham and 2e Ham in AO basis, then transform that to MO basis.
    At the same time, you can compute the core energy.
    '''
    cell = kmf.cell
    nkpts = len(kmf.kpts)
    nao = cell.nao_nr()
    energy_nuc = cell.energy_nuc()
    mo_coeff = np.asarray(mo_coeff)
    mo_cas = [mo_coeff[k, :, ncore:ncore+ncas] for k in range(nkpts)]
    mo_core = [mo_coeff[k, :, :ncore] for k in range(nkpts)]

    # Get the one-electron integrals
    hcore = kmf.get_hcore() 
    coredm = get_coredm(mo_core, nkpts)
    veff = kmf.get_veff(cell, dm_kpts=coredm)

    # Core energy
    Fpq = hcore + 0.5 * veff
    ecore = 0
    ecore = sum(np.einsum('ij,ji', coredm[k], Fpq[k]) for k in range(nkpts))

    # ecore = sum(np.einsum('ij,ji', dm, F) for dm, F in zip(coredm, Fpq))

    # for k in range(nkpts):
    #     ecore += np.einsum('ij, ji', coredm[k],Fpq[k])
    
    ecore += nkpts*energy_nuc 
    
    # Remember the nuclear repulsion energy is for the unit-cell but FCI problem 
    # is solved for the entire system. In the final step, I will divide the total energy 
    # by the nkpts to get the energy per unit cell.

    # Transform the integrals to MO basis in k-space then transform it to real space
    h1ao_k = hcore + veff

    scell, phase, mo_coeff_R, mo_phase = get_mo_coeff_k2R(kmf, mo_coeff, ncore, ncas)

    h1ao_R = np.einsum('Rk,kij,Sk->RiSj', phase, h1ao_k, phase.conj())
    h1ao_R = h1ao_R.reshape(nkpts*nao, nkpts*nao)
    h1mo_R = _basis_transformation(h1ao_R, mo_coeff_R)

    eri = active_space_eri(scell, mo_coeff_R, eri_file)
    eris = eri.reshape(nkpts*ncas, nkpts*ncas, nkpts*ncas, nkpts*ncas)

    coredm, h1ao_R, veff, hcore = None, None, None, None  # Free memory

    # Orthonormality check
    ovlp = scell.pbc_intor('int1e_ovlp')
    S = _basis_transformation(ovlp, mo_coeff_R)
    I = np.eye(nkpts*ncas)
    err = np.max(np.abs(S - I))
    assert err < 1e-8, "Max deviation from orthonormality in CAS MO basis:" + str(err)
    
    return ecore, h1mo_R, eris, mo_coeff_R

nkpts = len(kmf.kpts)
ncore = 6
nelecas  = 2
ncas = 2

eri_file = f'PAchain.{nkpts}.real.gdf'

h0, h1, h2, mo_coeff_R = get_fci_ham(kmf, ncore, ncas, mo_coeff, eri_file)

# h1 = 0.5*(h1 + h1.T.conj())
# h2 = 0.5*(h2 + h2.transpose(2, 3, 0, 1))
# h2 = 0.5*(h2 + h2.transpose(1, 0, 3, 2).conj())
# h2 = 0.5*(h2 + h2.transpose(3, 2, 1, 0).conj())

cisolver = direct_com_real.FCISolver()

e_tot_fromFCI, fcivec = cisolver.kernel(h1, h2, nkpts*ncas, (nkpts*1,nkpts*1), ecore=h0, verbose=4)
fcivec /= np.sqrt(np.vdot(fcivec, fcivec))
print("FCI - SCF Energy:", e_tot_fromFCI/nkpts - kmf.e_tot )

# # Compare the energy computations from different methods:
# # Test-1: Compute RDM1 and RDM2 and contract with the Hamiltonian
# rdm1, rdm2 = cisolver.make_rdm12_py(fcivec, nkpts*ncas, (nkpts*1,nkpts*1), reorder=True)
# e_tot_fromRDM = np.einsum('pq,qp', h1, rdm1) + 0.5* lib.einsum('pqrs,pqrs', h2, rdm2)
# e_tot_fromRDM += h0

# # Test-2: Based on the energy routine
# e_tot_fromEnergy = cisolver.energy(h1, h2, fcivec, nkpts*ncas, (nkpts*1,nkpts*1))
# e_tot_fromEnergy += h0

# print("FCI energy difference (RDMs based - kernel):", (e_tot_fromRDM - cisolver.eci).real / nkpts)
# print("FCI energy difference (Energy routine based - kernel)", (e_tot_fromEnergy - cisolver.eci).real / nkpts)

# Computing the RDM1s using the C solver
# def compare_rdm1s(rdm1, rdm1ref):
#     diff = np.max(np.abs(rdm1 - rdm1ref))
#     print("Max difference in RDM1s:", diff)

# rdm1a, rdm1b = cisolver.make_rdm1s(fcivec, nkpts*ncas, (nkpts*1,nkpts*1))
# rdm1aref, rdm1bref = cisolver.make_rdm1s_py(fcivec, nkpts*ncas, (nkpts*1,nkpts*1))
# rdm1check = cisolver.make_rdm12s_py(fcivec, nkpts*ncas, (nkpts*1,nkpts*1))[0]
# compare_rdm1s(rdm1check[0], rdm1aref)
# compare_rdm1s(rdm1check[1], rdm1bref)
# compare_rdm1s(rdm1a, rdm1aref)
# compare_rdm1s(rdm1b, rdm1bref)
# compare_rdm1s(rdm1a+rdm1b, rdm1bref+rdm1bref)




# Time to compare the 2-RDMs

def compare_rdm2s(rdm2, rdm2ref):
    diff = np.max(np.abs(rdm2 - rdm2ref))
    print("Max difference in RDM2s:", diff)
    print("Max difference in real part of RDM2s:", np.max(np.abs(rdm2.real - rdm2ref.real)))
    print("Max difference in imag part of RDM2s:", np.max(np.abs(rdm2.imag - rdm2ref.imag)))

rdm2aa, rdm2ab, rdm2bb = cisolver.make_rdm12s(fcivec, nkpts*ncas, (nkpts*1,nkpts*1))[1]
rdm2aa_c, rdm2ab_c, rdm2bb_c = cisolver.make_rdm12s_py(fcivec, nkpts*ncas, (nkpts*1,nkpts*1))[1]
compare_rdm2s(rdm2aa, rdm2aa_c)
compare_rdm2s(rdm2ab, rdm2ab_c)
compare_rdm2s(rdm2bb, rdm2bb_c)