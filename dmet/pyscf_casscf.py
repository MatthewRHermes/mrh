'''
pyscf-CASSCF SOLVER for DMET

To use this solver, one need to modify the dmet module to recognize the pyscf_casscf.
Specifically:
line 33: assert (( method == 'ED' ) or ( method == 'CC' ) or ( method == 'MP2' ) or ( method == 'CASSCF' ))
line 257-261:
elif ( self.method == 'CASSCF' ):
    import pyscf_casscf
    assert( Nelec_in_imp % 2 == 0 )
    DMguessRHF = self.ints.dmet_init_guess_rhf( loc2dmet, Norb_in_imp, Nelec_in_imp//2, numImpOrbs, chempot_imp )
    IMP_energy, IMP_1RDM = pyscf_casscf.solve( 0.0, dmetOEI, dmetFOCK, dmetTEI, Norb_in_imp, Nelec_in_imp, numImpOrbs, DMguessRHF, chempot_imp )

History: 

- the solver is tested under FCI limit. The energy agrees with the FCI energy by chemps2 solver.
However, the energy is explosive when the active space decreasing. VERY negative! => SOLVED

- Need to improve the efficiency => SOLVED

author: Hung Pham (email: phamx494@umn.edu)
'''

import numpy as np
import localintegrals
import os, time
import sys
#import qcdmet_paths
from pyscf import gto, scf, ao2mo, mcscf
#np.set_printoptions(threshold=np.nan)

def solve( CONST, OEI, FOCK, TEI, Norb, Nel, Nimp, impCAS, cas_list, DMguessRHF, energytype='CASCI', chempot_imp=0.0, printoutput=True ):

    # Augment the FOCK operator with the chemical potential
    FOCKcopy = FOCK.copy()
    if (chempot_imp != 0.0):
        for orb in range(Nimp):
            FOCKcopy[ orb, orb ] -= chempot_imp
    
    # Get the RHF solution
    mol = gto.Mole()
    mol.build(verbose=0)
    mol.atom.append(('H', (0, 0, 0)))
    mol.nelectron = Nel
    #mol.incore_anyway = True
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: FOCKcopy
    mf.get_ovlp = lambda *args: np.eye(Norb)
    mf._eri = ao2mo.restore(8, TEI, Norb)
    mf.scf()
    MOmf = mf.mo_coeff
    #print(mf.mo_occ)	
    '''	
    idx = mf.mo_energy.argsort()
    mf.mo_energy = mf.mo_energy[idx]
    mf.mo_coeff = mf.mo_coeff[:,idx]'''

    # Get the CASSCF solution
    CASe = impCAS[0]
    CASorb = impCAS[1]	
    checkCAS =  (CASe <= Nel) and (CASorb <= Norb)
    if (checkCAS == False):
        CASe = Nel
        CASorb = Norb
    mc = mcscf.CASSCF(mf, CASorb, CASe)	
    #mc.natorb = True
    if cas_list is not None: 
        print('Impurity active space selection:', cas_list)
        mo = mc.sort_mo(cas_list)
        E_CASSCF = mc.kernel(mo)[0]
    else:
        E_CASSCF = mc.kernel()[0]
    MO = mc.mo_coeff #save the MO coefficient
    MOnat = mc.cas_natorb()[0]
    OccNum = mc.cas_natorb()[2]	
    print('Dimension:', MO.shape[0] )	
    print('Impurity active space: ', CASe, 'electrons in ', CASorb, ' orbitals')	
    print('Impurity CASSCF energy: ', E_CASSCF)	
    #print('CASSCF orbital:', mc.mo_energy)	
    #print('NATURAL ORBITAL:')	
    #mc.analyze()
	
    # Get TwoRDM + OneRDM
    Norbcas = mc.ncas
    Norbcore = mc.ncore
    Nelcas = mc.nelecas	

    mocore = mc.mo_coeff[:,:Norbcore]
    mocas = mc.mo_coeff[:,Norbcore:Norbcore+Norbcas]

	
    casdm1 = mc.fcisolver.make_rdm1(mc.ci, Norbcas, Nelcas) #in CAS space
	# Transform the casdm2 (in CAS space) to casdm2lo (localized space).     
    # Dumb and lazy way: casdm1lo = np.einsum('pq,ap,bq->ab', casdm1, mocas, mocas) #in localized space
    casdm1lo = np.einsum('ap,pq->aq', mocas, casdm1)
    casdm1lo = np.einsum('bq,aq->ab', mocas, casdm1lo)
    coredm1 = np.dot(mocore, mocore.T) * 2 #in localized space
    OneRDM = coredm1 + casdm1lo	

    casdm2 = mc.fcisolver.make_rdm2(mc.ci,Norbcas,Nelcas) #in CAS space
	# Transform the casdm2 (in CAS space) to casdm2lo (localized space). 
	# Dumb and lazy way: casdm2ao = np.einsum('pqrs,ap,bq,cr,ds->abcd',casdm2,mocas,mocas,mocas,mocas)
    casdm2lo = np.einsum('ap,pqrs->aqrs', mocas, casdm2)
    casdm2lo = np.einsum('bq,aqrs->abrs', mocas, casdm2lo)
    casdm2lo = np.einsum('cr,abrs->abcs', mocas, casdm2lo)
    casdm2lo = np.einsum('ds,abcs->abcd', mocas, casdm2lo)	
	
    coredm2 = np.zeros([Norb, Norb, Norb, Norb]) #in AO
    coredm2 += np.einsum('pq,rs-> pqrs',coredm1,coredm1)
    coredm2 -= 0.5*np.einsum('ps,rq-> pqrs',coredm1,coredm1)
	
    effdm2 = np.zeros([Norb, Norb, Norb, Norb]) #in AO
    effdm2 += 2*np.einsum('pq,rs-> pqrs',casdm1lo,coredm1)
    effdm2 -= np.einsum('ps,rq-> pqrs',casdm1lo,coredm1)				
					
    TwoRDM = coredm2 + casdm2lo + effdm2	
    # To calculate the impurity energy, rescale the JK matrix with a factor 0.5 to avoid double counting: 0.5 * ( OEI + FOCK ) = OEI + 0.5 * JK
    '''
    This is the equation taken from the chemps2 solver:
    ImpurityEnergy = CONST
    ImpurityEnergy += 0.5 * np.einsum( 'ij,ij->', OneRDM[:Nimp,:], OEI[:Nimp,:] + FOCK[:Nimp,:] )
    ImpurityEnergy += 0.5 * np.einsum( 'ijkl,ijkl->', TwoRDM[:Nimp,:,:,:], TEI[:Nimp,:,:,:] )'''
    #This is the equation taken from the cc solver:
    if ( energytype == 'CASCI' ):
        ImpurityEnergy = E_CASSCF
    else:		
        ImpurityEnergy = CONST \
                       + 0.50  * np.einsum('ij,ij->',     OneRDM[:Nimp,:],     FOCK[:Nimp,:] + OEI[:Nimp,:]) \
                       + 0.125 * np.einsum('ijkl,ijkl->', TwoRDM[:Nimp,:,:,:], TEI[:Nimp,:,:,:]) \
                       + 0.125 * np.einsum('ijkl,ijkl->', TwoRDM[:,:Nimp,:,:], TEI[:,:Nimp,:,:]) \
                       + 0.125 * np.einsum('ijkl,ijkl->', TwoRDM[:,:,:Nimp,:], TEI[:,:,:Nimp,:]) \
                       + 0.125 * np.einsum('ijkl,ijkl->', TwoRDM[:,:,:,:Nimp], TEI[:,:,:,:Nimp])
    return ( ImpurityEnergy, OneRDM, MOmf, MO, MOnat, OccNum)