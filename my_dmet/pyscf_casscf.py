'''
pyscf-CASSCF SOLVER for DMET

To use this solver, one need to modify the dmet module to recognize the pyscf_mcascf.
Specifically:
line 33: assert (( method == 'ED' ) or ( method == 'CC' ) or ( method == 'MP2' ) or ( method == 'CASSCF' ))
line 257-261:
elif ( self.method == 'CASSCF' ):
    import pyscf_mcascf
    assert( Nelec_in_imp % 2 == 0 )
    guess_1RDM = self.ints.dmet_init_guess_rhf( loc2dmet, Norb_in_imp, Nelec_in_imp//2, numImpOrbs, chempot_frag )
    IMP_energy, IMP_1RDM = pyscf_mcascf.solve( 0.0, dmetOEI, dmetFOCK, dmetTEI, Norb_in_imp, Nelec_in_imp, numImpOrbs, guess_1RDM, chempot_frag )

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
from mrh.util.basis import represent_operator_in_basis

#def solve( CONST, OEI, FOCK, TEI, frag.norbs_imp, frag.nelec_imp, frag.norbs_frag, impCAS, frag.active_orb_list, guess_1RDM, energytype='CASCI', chempot_frag=0.0, printoutput=True ):
def solve (frag, guess_1RDM, chempot_frag=0.0):

    # Augment the FOCK operator with the chemical potential
    FOCKcopy = frag.impham_FOCK.copy()
    if (chempot_frag != 0.0):
        for orb in range(frag.norbs_frag):
            FOCKcopy[ orb, orb ] -= chempot_frag
    
    # Get the RHF solution
    mol = gto.Mole()
    mol.build(verbose=0)
    mol.atom.append(('H', (0, 0, 0)))
    mol.nelectron = frag.nelec_imp
    #mol.incore_anyway = True
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: FOCKcopy
    mf.get_ovlp = lambda *args: np.eye(frag.norbs_imp)
    mf._eri = ao2mo.restore(8, frag.impham_TEI, frag.norbs_imp)
    mf.scf(guess_1RDM)
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
    imp2mcno = mc.cas_natorb()[0]
    frag.loc2mcno = np.dot (frag.loc2imp, imp2mcno)
    frag.mcno_evals = mc.cas_natorb()[2]    
    #print('Dimension:', MO.shape[0] )    
    #print('Impurity active space: ', CASe, 'electrons in ', CASorb, ' orbitals')    
    #print('Impurity CASSCF energy: ', E_CASSCF)    
    #print('CASSCF orbital:', mc.mo_energy)    
    #print('NATURAL ORBITAL:')    
    #mc.analyze()
    
    # Get twoRDM + oneRDM. mcc: MC-SCF core, mca: MC-SCF active space
    # I'm going to need to keep some representation of the active-space orbitals
    norbs_mca = mc.ncas
    norbs_mcc = mc.ncore
    nelec_mca = mc.nelecas    

    imp2mcc = mc.mo_coeff[:,:norbs_mcc]
    imp2mca = mc.mo_coeff[:,norbs_mcc:norbs_mcc+norbs_mca]
    imp2mcno = mc.cas_natorb()[0][:,norbs_mcc:norbs_mcc+norbs_mca]
    frag.loc2mcno = np.dot (frag.loc2imp, imp2mcno)
    frag.mcno_evals = mc.cas_natorb()[2][norbs_mcc:norbs_mcc+norbs_mca]

    # MC-core oneRDM and twoRDM 
    oneRDMmcc_imp = np.dot(imp2mcc, imp2mcc.T) * 2 
    twoRDMmcc_imp = np.zeros([frag.norbs_imp, frag.norbs_imp, frag.norbs_imp, frag.norbs_imp]) 
    twoRDMmcc_imp +=     np.einsum('pq,rs-> pqrs',oneRDMmcc_imp,oneRDMmcc_imp)
    twoRDMmcc_imp -= 0.5*np.einsum('ps,rq-> pqrs',oneRDMmcc_imp,oneRDMmcc_imp)

    # MC-active oneRDM 
    oneRDMmca_mca = mc.fcisolver.make_rdm1(mc.ci, norbs_mca, nelec_mca)
    oneRDMmca_imp = np.einsum('ap,pq->aq', imp2mca, oneRDMmca_mca)
    oneRDMmca_imp = np.einsum('bq,aq->ab', imp2mca, oneRDMmca_imp)

    # MC-active twoRDM
    twoRDMmca_mca = mc.fcisolver.make_rdm2(mc.ci,norbs_mca,nelec_mca) #in CAS space
    twoRDMmca_imp = np.einsum('ap,pqrs->aqrs', imp2mca, twoRDMmca_mca)
    twoRDMmca_imp = np.einsum('bq,aqrs->abrs', imp2mca, twoRDMmca_imp)
    twoRDMmca_imp = np.einsum('cr,abrs->abcs', imp2mca, twoRDMmca_imp)
    twoRDMmca_imp = np.einsum('ds,abcs->abcd', imp2mca, twoRDMmca_imp)    
    
    # MC-active/MC-core oneRDM cumulant-expansion twoRDM residual (pqrs = pq*rs - ps*qr) (Mulliken/chemist's notation!)
    # Note the abuse of symmetry!
    twoRDMres_imp = np.zeros([frag.norbs_imp, frag.norbs_imp, frag.norbs_imp, frag.norbs_imp])
    twoRDMres_imp += 2*np.einsum('pq,rs-> pqrs',oneRDMmca_imp,oneRDMmcc_imp)
    twoRDMres_imp -=   np.einsum('ps,rq-> pqrs',oneRDMmca_imp,oneRDMmcc_imp)

    frag.oneRDM_imp = oneRDMmcc_imp + oneRDMmca_imp    
    frag.twoRDM_imp = twoRDMmcc_imp + twoRDMmca_imp + twoRDMres_imp    
    # To calculate the impurity energy, rescale the JK matrix with a factor 0.5 to avoid double counting: 0.5 * ( OEI + FOCK ) = OEI + 0.5 * JK
    '''
    This is the equation taken from the chemps2 solver:
    ImpurityEnergy = frag.CONST
    ImpurityEnergy += 0.5 * np.einsum( 'ij,ij->', oneRDM[:frag.norbs_frag,:], OEI[:frag.norbs_frag,:] + FOCK[:frag.norbs_frag,:] )
    ImpurityEnergy += 0.5 * np.einsum( 'ijkl,ijkl->', twoRDM[:frag.norbs_frag,:,:,:], TEI[:frag.norbs_frag,:,:,:] )'''
    #This is the equation taken from the cc solver:
    frag.E_imp  = E_CASSCF
    frag.E_frag = 0.50   * np.einsum('ij,ij->',     frag.oneRDM_imp[:frag.norbs_frag,:],     frag.impham_FOCK[:frag.norbs_frag,:] + frag.impham_OEI[:frag.norbs_frag,:]) \
                 + 0.125 * np.einsum('ijkl,ijkl->', frag.twoRDM_imp[:frag.norbs_frag,:,:,:], frag.impham_TEI[:frag.norbs_frag,:,:,:]) \
                 + 0.125 * np.einsum('ijkl,ijkl->', frag.twoRDM_imp[:,:frag.norbs_frag,:,:], frag.impham_TEI[:,:frag.norbs_frag,:,:]) \
                 + 0.125 * np.einsum('ijkl,ijkl->', frag.twoRDM_imp[:,:,:frag.norbs_frag,:], frag.impham_TEI[:,:,:frag.norbs_frag,:]) \
                 + 0.125 * np.einsum('ijkl,ijkl->', frag.twoRDM_imp[:,:,:,:frag.norbs_frag], frag.impham_TEI[:,:,:,:frag.norbs_frag])

    return None


