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
from . import localintegrals
import os, time
import sys
#import qcdmet_paths
from pyscf import gto, scf, ao2mo, mcscf, ao2mo
from pyscf.tools import molden
#np.set_printoptions(threshold=np.nan)
from mrh.util.basis import represent_operator_in_basis, project_operator_into_subspace, orthonormalize_a_basis, get_complete_basis
from mrh.util.rdm import get_2CDM_from_2RDM
from mrh.util.tensors import symmetrize_tensor
from functools import reduce

#def solve( CONST, OEI, FOCK, TEI, frag.norbs_imp, frag.nelec_imp, frag.norbs_frag, impCAS, frag.active_orb_list, guess_1RDM, energytype='CASCI', chempot_frag=0.0, printoutput=True ):
def solve (frag, guess_1RDM, chempot_imp):

    # Augment OEI with the chemical potential
    OEI = frag.impham_OEI - chempot_imp
    
    # Get the RHF solution
    mol = gto.Mole()
    mol.build(verbose=0)
    mol.atom.append(('H', (0, 0, 0)))
    mol.nelectron = frag.nelec_imp
    #mol.incore_anyway = True
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: OEI
    mf.get_ovlp = lambda *args: np.eye(frag.norbs_imp)
    mf._eri = ao2mo.restore(8, frag.impham_TEI, frag.norbs_imp)
    mf.scf(guess_1RDM)
    print ("CASSCF RHF-step energy: {0}".format (mf.e_tot))
    #print(mf.mo_occ)    
    '''    
    idx = mf.mo_energy.argsort()
    mf.mo_energy = mf.mo_energy[idx]
    mf.mo_coeff = mf.mo_coeff[:,idx]'''

    # If I haven't yet, print out the MOs so I can pick a good active space
    if frag.mfmo_printed == False:
        imp2mo = mf.mo_coeff
        frag.loc2mo = np.dot (frag.loc2imp, imp2mo)
        frag.impurity_molden ('init_HF')
        frag.mfmo_printed = True
        
    # Get the CASSCF solution
    CASe = frag.active_space[0]
    CASorb = frag.active_space[1]    
    checkCAS =  (CASe <= frag.nelec_imp) and (CASorb <= frag.norbs_imp)
    if (checkCAS == False):
        CASe = frag.nelec_imp
        CASorb = frag.norbs_imp
    mc = mcscf.CASSCF(mf, CASorb, CASe)    
    #mc.natorb = True

    # Guess orbitals
    if frag.norbs_as > 0:
        print ("Projecting stored active-space mos (frag.loc2amo) onto the impurity basis")
        imp2mo = orthonormalize_a_basis (frag.imp2amo)
        assert (imp2mo.shape[1] == CASorb)
        imp2mo = get_complete_basis (imp2mo)
        imp2mo = mc.sort_mo (list(range(frag.norbs_as)), mo_coeff=imp2mo, base=0)
        assert (imp2mo.shape == (frag.norbs_imp, frag.norbs_imp))
    elif np.prod (frag.active_orb_list.shape) > 0: 
        print('Impurity active space selection:', frag.active_orb_list)
        imp2mo = mc.sort_mo(frag.active_orb_list)
    else:
        imp2mo = mc.mo_coeff 
    E_CASSCF = mc.kernel(imp2mo)[0]
    E_CASSCF = mc.kernel()[0] # Because the convergence checker is sometimes bad
    print('Impurity CASSCF energy (incl chempot): ', frag.impham_CONST + E_CASSCF)    
    
    # Get twoRDM + oneRDM. cs: MC-SCF core, as: MC-SCF active space
    # I'm going to need to keep some representation of the active-space orbitals
    norbs_amo = mc.ncas
    norbs_cmo = mc.ncore
    nelec_amo = mc.nelecas    
    imp2mo = mc.mo_coeff #mc.cas_natorb()[0]
    frag.loc2mo = np.dot (frag.loc2imp, imp2mo)
    frag.loc2amo = np.copy (frag.loc2mo[:,norbs_cmo:norbs_cmo+norbs_amo])

    # oneRDM
    oneRDM_imp = mc.make_rdm1 ()

    # twoCDM
    oneRDM_amo, twoRDM_amo = mc.fcisolver.make_rdm12 (mc.ci, norbs_amo, nelec_amo)
    twoCDM_amo = get_2CDM_from_2RDM (twoRDM_amo, oneRDM_amo)
    twoCDM_imp = represent_operator_in_basis (twoCDM_amo, frag.amo2imp)

    # General impurity data
    frag.oneRDM_loc = symmetrize_tensor (frag.oneRDMfroz_loc + represent_operator_in_basis (oneRDM_imp, frag.imp2loc))
    frag.twoCDM_imp = symmetrize_tensor (twoCDM_imp)
    frag.E_imp      = frag.impham_CONST + E_CASSCF + np.einsum ('ab,ab->', chempot_imp, oneRDM_imp)

    # Active-space RDM data
    frag.oneRDMas_loc  = symmetrize_tensor (represent_operator_in_basis (oneRDM_amo, frag.amo2loc))
    frag.twoCDMimp_amo = twoCDM_amo

    return None


