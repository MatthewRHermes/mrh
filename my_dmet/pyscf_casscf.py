'''
pyscf-CASSCF SOLVER for DMET

To use this solver, one need to modify the dmet module to recognize the pyscf_amoscf.
Specifically:
line 33: assert (( method == 'ED' ) or ( method == 'CC' ) or ( method == 'MP2' ) or ( method == 'CASSCF' ))
line 257-261:
elif ( self.method == 'CASSCF' ):
    import pyscf_amoscf
    assert( Nelec_in_imp % 2 == 0 )
    guess_1RDM = self.ints.dmet_init_guess_rhf( loc2dmet, Norb_in_imp, Nelec_in_imp//2, numImpOrbs, chempot_frag )
    IMP_energy, IMP_1RDM = pyscf_amoscf.solve( 0.0, dmetOEI, dmetFOCK, dmetTEI, Norb_in_imp, Nelec_in_imp, numImpOrbs, guess_1RDM, chempot_frag )

History: 

- the solver is tested under FCI limit. The energy agrees with the FCI energy by chemps2 solver.
However, the energy is explosive when the active space decreasing. VERY negative! => SOLVED

- Need to improve the efficiency => SOLVED

author: Hung Pham (email: phamx494@umn.edu)
'''

import numpy as np
from mrh.my_dmet import localintegrals
import os, time
import sys
#import qcdmet_paths
from pyscf import gto, scf, ao2mo, mcscf, fci
from pyscf.mcscf.addons import spin_square
from pyscf.tools import molden
#np.set_printoptions(threshold=np.nan)
from mrh.util.la import matrix_eigen_control_options
from mrh.util.basis import represent_operator_in_basis, project_operator_into_subspace, orthonormalize_a_basis, get_complete_basis, get_complementary_states
from mrh.util.basis import is_basis_orthonormal, get_overlapping_states, is_basis_orthonormal_and_complete, compute_nelec_in_subspace
from mrh.util.rdm import get_2CDM_from_2RDM, get_2RDM_from_2CDM
from mrh.util.tensors import symmetrize_tensor
from mrh.my_pyscf import mcscf as my_mcscf
from mrh.my_pyscf.scf import hf_as
from functools import reduce

#def solve( CONST, OEI, FOCK, TEI, frag.norbs_imp, frag.nelec_imp, frag.norbs_frag, impCAS, frag.active_orb_list, guess_1RDM, energytype='CASCI', chempot_frag=0.0, printoutput=True ):
def solve (frag, guess_1RDM, chempot_imp):

    # Augment OEI with the chemical potential
    OEI = frag.impham_OEI - chempot_imp
    
    # Get the RHF solution
    mol = gto.Mole()
    mol.spin = 2*frag.spin_MS
    mol.verbose = 0 if frag.mol_output is None else 4
    mol.output = frag.mol_output
    mol.build ()
    mol.atom.append(('H', (0, 0, 0)))
    mol.nelectron = frag.nelec_imp
    #mol.incore_anyway = True
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: OEI
    mf.get_ovlp = lambda *args: np.eye(frag.norbs_imp)
    mf._eri = ao2mo.restore(8, frag.impham_TEI, frag.norbs_imp)
    mf.scf (guess_1RDM)
    if not mf.converged:
        print ("CASSCF RHF-step not converged on fixed-point iteration; initiating newton solver")
        mf = mf.newton ()
        mf.kernel ()

    # Instability check and repeat
    for i in range (frag.num_mf_stab_checks):
        mf.mo_coeff = mf.stability ()[0]
        guess_1RDM = mf.make_rdm1 ()
        mf = scf.RHF(mol)
        mf.get_hcore = lambda *args: OEI
        mf.get_ovlp = lambda *args: np.eye(frag.norbs_imp)
        mf._eri = ao2mo.restore(8, frag.impham_TEI, frag.norbs_imp)
        mf.scf (guess_1RDM)
        if not mf.converged:
            mf = mf.newton ()
            mf.kernel ()
    
    print ("CASSCF RHF-step energy: {} (elec: {})".format (mf.e_tot + frag.impham_CONST, mf.e_tot))
    #print(mf.mo_occ)    
    '''    
    idx = mf.mo_energy.argsort()
    mf.mo_energy = mf.mo_energy[idx]
    mf.mo_coeff = mf.mo_coeff[:,idx]'''

    # If I haven't yet, print out the MOs so I can pick a good active space
    if frag.mfmo_printed == False:
        ao2mfmo = reduce (np.dot, [frag.ints.ao2loc, frag.loc2imp, mf.mo_coeff])
        molden.from_mo (frag.ints.mol, frag.frag_name + '_mfmorb.molden', ao2mfmo, occ=mf.mo_occ, ene=mf.mo_energy)
        frag.mfmo_printed = True
        
    # Get the CASSCF solution
    CASe = frag.active_space[0]
    CASorb = frag.active_space[1]    
    checkCAS =  (CASe <= frag.nelec_imp) and (CASorb <= frag.norbs_imp)
    if (checkCAS == False):
        CASe = frag.nelec_imp
        CASorb = frag.norbs_imp
    mc = mcscf.CASSCF(mf, CASorb, CASe)
    norbs_amo = mc.ncas
    norbs_cmo = mc.ncore
    norbs_imo = frag.norbs_imp - norbs_amo
    nelec_amo = sum (mc.nelecas)
    norbs_occ = norbs_amo + norbs_cmo
    #mc.natorb = True

    # Guess orbitals
    if frag.norbs_as > 0:
        #if frag.loc2mo.shape[1] == frag.norbs_imp:
        #    print ("Projecting stored mos (frag.loc2mo) onto the impurity basis")
        #    imp2mo = mcscf.addons.project_init_guess (mc, np.dot (frag.imp2loc, frag.loc2mo))
        #else:
        print ("Projecting stored amos (frag.loc2amo) onto the impurity basis")
        imp2mo = project_amo_manually (frag.loc2imp, frag.loc2amo, mf.get_fock (), norbs_cmo)
        make_guess_molden (frag, frag.frag_name + '_orbiter.molden', imp2mo, norbs_cmo, norbs_amo)
    elif frag.loc2amo_guess.shape[1] > 0:
        print ("Projecting provided guess orbitals onto the impurity basis")
        imp2mo = project_amo_manually (frag.loc2imp, frag.loc2amo_guess, mf.get_fock (), norbs_cmo)
        frag.loc2amo_guess = np.zeros ((frag.norbs_tot, 0))
        #imp2mo = mcscf.addons.project_init_guess (mc, imp2mo)
        make_guess_molden (frag, frag.frag_name + '_guess.molden', imp2mo, norbs_cmo, norbs_amo)
    elif np.prod (frag.active_orb_list.shape) > 0: 
        print('Impurity active space selection:', frag.active_orb_list)
        imp2mo = mc.sort_mo(frag.active_orb_list)
    else:
        imp2mo = mc.mo_coeff 
        print ("Default impurity active space selection: {}".format (np.arange (norbs_cmo, norbs_occ, 1, dtype=int)))
    t_start = time.time()
    mc.fcisolver = fci.solver (mf.mol, singlet=(frag.spin_S == 0))
    if frag.spin_S != 0:
        s2_eval = frag.spin_S * (frag.spin_S + 1)
        mc.fix_spin_(ss=s2_eval)
    mc.ah_start_tol = 1e-8
    E_CASSCF = mc.kernel(imp2mo)[0]
    mc.conv_tol = 1e-12
    mc.ah_start_tol = 1e-10
    mc.ah_conv_tol = 1e-12
    E_CASSCF = mc.kernel()[0]
    if not mc.converged:
        mc = mc.newton ()
        E_CASSCF = mc.kernel()[0]
    assert (mc.converged)
    t_end = time.time()
    print('Impurity CASSCF energy (incl chempot): {}; spin multiplicity: {}; time to solve: {}'.format (frag.impham_CONST + E_CASSCF, spin_square (mc)[1], t_end - t_start))
    
    # Get twoRDM + oneRDM. cs: MC-SCF core, as: MC-SCF active space
    # I'm going to need to keep some representation of the active-space orbitals
    imp2mo = np.copy (mc.mo_coeff) #mc.cas_natorb()[0]
    loc2mo = np.dot (frag.loc2imp, imp2mo)
    imp2amo = imp2mo[:,norbs_cmo:norbs_occ]
    loc2amo = loc2mo[:,norbs_cmo:norbs_occ]

    # oneRDM
    oneRDM_imp = mc.make_rdm1 ()

    # twoCDM
    oneRDM_amo, twoRDM_amo = mc.fcisolver.make_rdm12 (mc.ci, norbs_amo, nelec_amo)
    twoCDM_amo = get_2CDM_from_2RDM (twoRDM_amo, oneRDM_amo)
    twoCDM_imp = represent_operator_in_basis (twoCDM_amo, imp2amo.conjugate ().T)

    # General impurity data
    frag.oneRDM_loc = symmetrize_tensor (frag.oneRDMfroz_loc + represent_operator_in_basis (oneRDM_imp, frag.imp2loc))
    frag.twoCDM_imp = symmetrize_tensor (twoCDM_imp)
    frag.E_imp      = frag.impham_CONST + E_CASSCF + np.einsum ('ab,ab->', chempot_imp, oneRDM_imp)

    # Active-space RDM data
    if (frag.frag_constrained_casscf):
        oneRDM_amo, twoCDM_amo, _, loc2amo = get_fragcasscf (frag, mf, loc2mo)
    frag.oneRDMas_loc  = symmetrize_tensor (represent_operator_in_basis (oneRDM_amo, loc2amo.conjugate ().T))
    frag.twoCDMimp_amo = twoCDM_amo
    frag.loc2mo = loc2mo
    frag.loc2amo = loc2amo


    '''
    mol = frag.ints.mol.copy ()
    mol.nelectron = frag.nelec_imp
    no_coeffs, _, no_occ = mc.cas_natorb ()
    loc2no = reduce (np.dot, [frag.ints.ao2loc, frag.loc2imp, no_coeffs])
    molden.from_mo (mol, 'test_casscf.molden', loc2no, occ=no_occ)
    '''

    return None

def get_fragcasscf (frag, mf, loc2mo):

    norbs_amo = frag.active_space[1]
    nelec_amo = frag.active_space[0]
    mo = np.dot (frag.imp2loc, loc2mo)

    mf2 = hf_as.RHF(mf.mol)
    mf2.wo_coeff = np.eye (mf.get_ovlp ().shape[0])
    mf2.get_hcore = lambda *args: mf.get_hcore ()
    mf2.get_ovlp = lambda *args: mf.get_ovlp ()
    mf2._eri = mf._eri
    mf2.scf(mf.make_rdm1 ())

    mc = my_mcscf.constrCASSCF (mf2, norbs_amo, nelec_amo, cas_ao=list(range(frag.norbs_frag)))
    norbs_cmo = mc.ncore
    norbs_occ = norbs_cmo + norbs_amo
    t_start = time.time ()
    E_fragCASSCF = mc.kernel (mo)[0]
    E_fragCASSCF = mc.kernel ()[0]
    t_end = time.time ()
    assert (mc.converged)
    print('Impurity fragCASSCF energy (incl chempot): {0}; time to solve: {1}'.format (frag.impham_CONST + E_fragCASSCF, t_end - t_start))

    casci = mcscf.CASCI (mf2, norbs_amo, nelec_amo)
    E_testfragCASSCF = casci.kernel (mc.mo_coeff)[0]
    assert (abs (E_testfragCASSCF - E_fragCASSCF) < 1e-8), E_testfragCASSCF

    loc2mo = np.dot (frag.loc2imp, mc.mo_coeff)
    loc2amo = loc2mo[:,norbs_cmo:norbs_occ]

    '''
    mol = frag.ints.mol.copy ()
    mol.nelectron = frag.nelec_imp
    imp2no, _, no_occ = mc.cas_natorb ()
    loc2no = reduce (np.dot, [frag.ints.ao2loc, frag.loc2imp, imp2no])
    molden.from_mo (mol, 'test_fragcasscf.molden', loc2no, occ=no_occ)
    '''

    oneRDM_amo, twoRDM_amo = mc.fcisolver.make_rdm12 (mc.ci, norbs_amo, nelec_amo)
    twoCDM_amo = get_2CDM_from_2RDM (twoRDM_amo, oneRDM_amo)
    return oneRDM_amo, twoCDM_amo, loc2mo, loc2amo
    

def project_amo_manually (loc2imp, loc2amo, fock_mf, norbs_cmo):
    norbs_amo = loc2amo.shape[1]
    amo2imp = np.dot (loc2amo.conjugate ().T, loc2imp)
    proj = np.dot (amo2imp.conjugate ().T, amo2imp)
    evals, evecs = matrix_eigen_control_options (proj, sort_vecs=-1, only_nonzero_vals=False)
    imp2amo = np.copy (evecs[:,:norbs_amo])
    imp2imo = np.copy (evecs[:,norbs_amo:])
    _, evecs = matrix_eigen_control_options (represent_operator_in_basis (fock_mf, imp2imo), sort_vecs=1, only_nonzero_vals=False)
    imp2imo = np.dot (imp2imo, evecs)
    imp2cmo = imp2imo[:,:norbs_cmo]
    imp2vmo = imp2imo[:,norbs_cmo:]
    return np.concatenate ([imp2cmo, imp2amo, imp2vmo], axis=1)

def make_guess_molden (frag, filename, imp2mo, norbs_cmo, norbs_amo):
    norbs_tot = imp2mo.shape[1]
    mo_occ = np.zeros (norbs_tot)
    norbs_occ = norbs_cmo + norbs_amo
    mo_occ[:norbs_cmo] = 2
    mo_occ[norbs_cmo:norbs_occ] = 1
    mo = reduce (np.dot, (frag.ints.ao2loc, frag.loc2imp, imp2mo))
    molden.from_mo (frag.ints.mol, filename, mo, occ=mo_occ)
    return

