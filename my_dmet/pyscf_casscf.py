'''
pyscf-CASSCF SOLVER for DMET

To use this solver, one need to modify the dmet module to recognize the pyscf_asscf.
Specifically:
line 33: assert (( method == 'ED' ) or ( method == 'CC' ) or ( method == 'MP2' ) or ( method == 'CASSCF' ))
line 257-261:
elif ( self.method == 'CASSCF' ):
    import pyscf_asscf
    assert( Nelec_in_imp % 2 == 0 )
    guess_1RDM = self.ints.dmet_init_guess_rhf( loc2dmet, Norb_in_imp, Nelec_in_imp//2, numImpOrbs, chempot_frag )
    IMP_energy, IMP_1RDM = pyscf_asscf.solve( 0.0, dmetOEI, dmetFOCK, dmetTEI, Norb_in_imp, Nelec_in_imp, numImpOrbs, guess_1RDM, chempot_frag )

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
from pyscf import gto, scf, ao2mo, mcscf
#np.set_printoptions(threshold=np.nan)
from mrh.util.basis import represent_operator_in_basis, project_operator_into_subspace
from mrh.util.rdm import get_2RDMR_from_2RDM

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

    # Get the CASSCF solution
    CASe = frag.active_space[0]
    CASorb = frag.active_space[1]    
    checkCAS =  (CASe <= frag.nelec_imp) and (CASorb <= frag.norbs_imp)
    if (checkCAS == False):
        CASe = frag.nelec_imp
        CASorb = frag.norbs_imp
    mc = mcscf.CASSCF(mf, CASorb, CASe)    
    #mc.natorb = True
    if np.prod (frag.active_orb_list.shape) > 0: 
        print('Impurity active space selection:', frag.active_orb_list)
        mo = mc.sort_mo(frag.active_orb_list)
        E_CASSCF = mc.kernel(mo)[0]
    else:
        E_CASSCF = mc.kernel()[0]
    #print('Dimension:', MO.shape[0] )    
    #print('Impurity active space: ', CASe, 'electrons in ', CASorb, ' orbitals')    
    print('Impurity CASSCF energy (incl chempot): ', frag.impham_CONST + E_CASSCF)    
    #print('CASSCF orbital:', mc.mo_energy)    
    #print('NATURAL ORBITAL:')    
    #mc.analyze()
    
    # Get twoRDM + oneRDM. cs: MC-SCF core, as: MC-SCF active space
    # I'm going to need to keep some representation of the active-space orbitals
    norbs_as = mc.ncas
    norbs_cs = mc.ncore
    nelec_as = mc.nelecas    
    imp2cs = mc.mo_coeff[:,:norbs_cs]
    imp2as = mc.mo_coeff[:,norbs_cs:norbs_cs+norbs_as]
    frag.loc2as = np.dot (frag.loc2imp, imp2as)

    # MC-core oneRDM 
    oneRDMcs_imp = project_operator_into_subspace (2.0 * np.eye (frag.norbs_imp), imp2cs) 

    # MC-active oneRDM 
    oneRDMas_as    = mc.fcisolver.make_rdm1(mc.ci, norbs_as, nelec_as)
    oneRDMas_imp   = represent_operator_in_basis (oneRDMas_as, frag.as2imp)
    oneRDMimp_imp  = oneRDMcs_imp + oneRDMas_imp

    # MC-active twoRDMR
    twoRDMimp_as   = mc.fcisolver.make_rdm2 (mc.ci, norbs_as, nelec_as)
    twoRDMRimp_as  = get_2RDMR_from_2RDM (mc.fcisolver.make_rdm2 (mc.ci, norbs_as, nelec_as), oneRDMas_as)
    twoRDMRimp_imp = represent_operator_in_basis (twoRDMRimp_as, frag.as2imp)
    '''
    twoRDMRimp_as  = mc.fcisolver.make_rdm2(mc.ci,norbs_as,nelec_as) #in CAS space
    twoRDMRimp_as -=     np.einsum ('pq,rs->pqrs', oneRDMas_as, oneRDMas_as)
    twoRDMRimp_as += 0.5*np.einsum ('ps,rq->pqrs', oneRDMas_as, oneRDMas_as)
    twoRDMRimp_imp = np.einsum('ap,pqrs->aqrs', imp2as, twoRDMRimp_as)
    twoRDMRimp_imp = np.einsum('bq,aqrs->abrs', imp2as, twoRDMRimp_imp)
    twoRDMRimp_imp = np.einsum('cr,abrs->abcs', imp2as, twoRDMRimp_imp)
    twoRDMRimp_imp = np.einsum('ds,abcs->abcd', imp2as, twoRDMRimp_imp)    
    '''

    # General impurity data
    frag.oneRDM_loc     = frag.oneRDMfroz_loc + represent_operator_in_basis (oneRDMimp_imp, frag.imp2loc)
    frag.twoRDMRimp_imp = twoRDMRimp_imp
    frag.E_imp          = frag.impham_CONST + E_CASSCF + np.einsum ('ab,ab->', chempot_imp, oneRDMimp_imp)

    # Active-space RDM data
    frag.oneRDMas_loc = represent_operator_in_basis (oneRDMas_imp, frag.imp2loc)
    frag.twoRDMas_imp = represent_operator_in_basis (twoRDMimp_as, frag.as2imp)
    frag.loc2imp_last = np.copy (frag.loc2imp)

    return None


