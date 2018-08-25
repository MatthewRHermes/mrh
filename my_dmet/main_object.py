'''
# This is going to be a complicated overlapping set of code with the out-of-date, no-longer-supported QC-DMET by Sebastian Wouters,
# Hung Pham's code, and my additions or modifications

-- Sebastian Wouters' copyright below

    QC-DMET: a python implementation of density matrix embedding theory for ab initio quantum chemistry
    Copyright (C) 2015 Sebastian Wouters
    
    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.
    
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
'''

from mrh.my_dmet import localintegrals, qcdmethelper
import numpy as np
from scipy import optimize, linalg
import time
#import tracemalloc
from pyscf.tools import molden
from mrh.util import params
from mrh.util.la import matrix_eigen_control_options
from mrh.util.rdm import S2_exptval
from mrh.util.basis import represent_operator_in_basis, orthonormalize_a_basis, get_complementary_states, project_operator_into_subspace, is_matrix_eye, measure_basis_olap, is_basis_orthonormal_and_complete, is_basis_orthonormal
from mrh.my_dmet.debug import debug_ofc_oneRDM, debug_Etot, examine_ifrag_olap, examine_wmcs
from functools import reduce
from itertools import combinations, product

class dmet:

    def __init__( self, theInts, fragments, isTranslationInvariant=False, SCmethod='BFGS', incl_bath_errvec=True, use_constrained_opt=False, 
                    doDET=False, doDET_NO=False, CC_E_TYPE='LAMBDA', minFunc='FOCK_INIT', wma_options=False, print_u=True,
                    print_rdm=True, mc_dmet=False, debug_energy=False, debug_reloc=False, noselfconsistent=False, do_constrCASSCF=False,
                    do_refragmentation=True, incl_impcore_correlation=False, nelec_int_thresh=1e-6, chempot_init=0.0, num_mf_stab_checks=0,
                    corrpot_maxiter=50, orb_maxiter=50, chempot_tol=1e-6, corrpot_mf_moldens=0):

        if isTranslationInvariant:
            raise RuntimeError ("The translational invariance option doesn't work!  It needs to be completely rebuilt!")
            assert( theInts.TI_OK == True )
            assert( len (fragments) == 1 )
        
        assert (( SCmethod == 'LSTSQ' ) or ( SCmethod == 'BFGS' ) or ( SCmethod == 'NONE' ))
        assert (( CC_E_TYPE == 'LAMBDA') or ( CC_E_TYPE == 'CASCI'))        

        #tracemalloc.start (10)

        self.ints                     = theInts
        self.norbs_tot                = self.ints.norbs_tot
        self.fragments                = fragments
        self.NI_hack                  = False
        self.doSCF                    = False
        self.TransInv                 = isTranslationInvariant
        self.doDET                    = doDET or doDET_NO
        self.doDET_NO                 = doDET_NO
        self.CC_E_TYPE                = CC_E_TYPE
        self.minFunc                  = minFunc
        self.wma_options              = wma_options
        self.print_u                  = print_u
        self.print_rdm                = print_rdm
        self.SCmethod                 = 'NONE' if self.doDET else SCmethod
        self.incl_bath_errvec         = False if self.doDET else incl_bath_errvec
        self.altcostfunc              = False if self.doDET else use_constrained_opt
        self.mc_dmet                  = mc_dmet
        self.debug_energy             = debug_energy
        self.debug_reloc              = debug_reloc
        self.noselfconsistent         = noselfconsistent
        self.do_constrCASSCF          = do_constrCASSCF
        self.do_refragmentation       = do_refragmentation
        self.incl_impcore_correlation = incl_impcore_correlation
        self.nelec_int_thresh         = nelec_int_thresh
        self.chempot_init             = chempot_init
        self.num_mf_stab_checks       = num_mf_stab_checks
        self.corrpot_maxiter          = corrpot_maxiter
        self.orb_maxiter              = orb_maxiter
        self.chempot_tol              = chempot_tol
        self.corrpot_mf_moldens       = corrpot_mf_moldens
        self.corrpot_mf_molden_cnt    = 0
        assert (np.any (np.array ([do_constrCASSCF, do_refragmentation]) == False)), 'constrCASSCF and refragmentation inconsistent with each other'

        if self.noselfconsistent:
            SCmethod = 'NONE'
        self.ints.num_mf_stab_checks = num_mf_stab_checks
        for frag in self.fragments:
            frag.debug_energy             = debug_energy
            frag.incl_impcore_correlation = incl_impcore_correlation
            frag.num_mf_stab_checks       = num_mf_stab_checks
        if self.doDET:
            print ("Note: doing DET overrides settings for SCmethod, incl_bath_errvec, and altcostfunc, all of which have only one value compatible with DET")
        self.examine_ifrag_olap = False
        self.examine_wmcs = False
        self.ofc_emb_init_ncycles = 3

        self.acceptable_errvec_check ()
        if self.altcostfunc:
            assert (self.SCmethod == 'BFGS' or self.SCmethod == 'NONE')

        self.get_allcore_orbs ()
        if ( self.norbs_allcore > 0 ): # One or more impurities which do not cover the entire system
            assert( self.TransInv == False ) # Make sure that you don't work translational invariant
            # Note on working with impurities which do no tile the entire system: they should be the first orbitals in the Hamiltonian!

        def umat_ftriu_mask (x, k=0):
            r = np.zeros (x.shape, dtype=np.bool)
            for frag in self.fragments:
                ftriu = np.triu_indices (frag.norbs_frag)
                c = tuple (np.asarray ([frag.frag_orb_list[i] for i in f]) for f in ftriu)
                r[c] = True
            return r
        self.umat_ftriu_idx = np.mask_indices (self.norbs_tot, umat_ftriu_mask)
        
        self.loc2fno    = None
        self.umat       = np.zeros([ self.norbs_tot, self.norbs_tot ])
        self.relaxation = 0.0
        self.energy     = 0.0
        self.spin       = 0.0
        self.mu_imp     = 0.0
        self.helper     = qcdmethelper.qcdmethelper( self.ints, self.makelist_H1(), self.altcostfunc, self.minFunc )
        
        np.set_printoptions(precision=3, linewidth=160)
        #objinit = tracemalloc.take_snapshot ()
        #objinit.dump ('objinit.snpsht')
        

    def acceptable_errvec_check (obj):
        errcheck = (
         ("bath in errvec incompatible with DET" ,             obj.incl_bath_errvec and obj.doDET),
         ("constrained opt incompatible with DET" ,            obj.altcostfunc and obj.doDET),
         ("bath in errvec incompatible with constrained opt" , obj.altcostfunc and obj.incl_bath_errvec),
         ("natural orbital basis not meaningful for DMET" ,    obj.doDET_NO and (not obj.doDET))
                   )
        for errstr, err in errcheck:
            if err:
                raise RuntimeError(errstr)


    def get_allcore_orbs ( self ):
        # I guess this just determines whether every orbital has a fragment or not    

        quicktest = np.zeros([ self.norbs_tot ], dtype=int)
        for frag in self.fragments:
            quicktest += np.abs(frag.is_frag_orb.astype (int))
        assert( np.all( quicktest >= 0 ) )
        assert( np.all( quicktest <= 1 ) )
        self.is_allcore_orb = np.logical_not (quicktest.astype (bool))
        self.oneRDMallcore_loc = np.zeros_like (self.ints.activeOEI)
            
    @property
    def loc2allcore (self):
        return np.eye (self.norbs_tot)[:,self.is_allcore_orb]

    @property
    def Pallcore (self):
        return np.diag (self.is_allcore_orb).astype (int)

    @property
    def norbs_allcore (self):
        return np.count_nonzero (self.is_allcore_orb)

    @property
    def allcore_orb_list (self):
        return np.flatnonzero (self.is_allcore_orb)

    @property
    def nelec_wma (self):
        return int (sum (frag.active_space[0] for frag in self.fragments))

    @property
    def norbs_wma (self):
        return int (sum (frag.active_space[1] for frag in self.fragments))

    @property
    def norbs_wmc (self):
        return self.norbs_tot - self.norbs_wma

    def makelist_H1( self ):
   
        # OK, this is somehow related to the C code that came with this that does rhf response. 
        theH1 = []
        if ( self.doDET == True ): # Do density embedding theory
            if ( self.TransInv == True ): # Translational invariance assumed
                # In this case, it appears that H1 identifies a set of diagonal 1RDM elements that are equivalent by symmetry
                for row in range( self.fragments[0].norbs_frag ):
                    H1 = np.zeros( [ self.norbs_tot, self.norbs_tot ], dtype=int )
                    for jumper in range( self.norbs_tot // self.fragments[0].norbs_frag ):
                        jumpsquare = self.fragments[0].norbs_frag * jumper
                        H1[ jumpsquare + row, jumpsquare + row ] = 1
                    theH1.append( H1 )
            else: # NO translational invariance assumed
                # Huh? In this case it's a long list of giant matrices with only one nonzero value each
                jumpsquare = 0
                for frag in self.fragments:
                    for row in range( frag.norbs_frag ):
                        H1 = np.zeros( [ self.norbs_tot, self.norbs_tot ], dtype=int )
                        H1[ jumpsquare + row, jumpsquare + row ] = 1
                        theH1.append( H1 )
                    jumpsquare += frag.norbs_frag
        else: # Do density MATRIX embedding theory
            # Upper triangular parts of 1RDMs only
            if ( self.TransInv == True ): # Translational invariance assumed
                # Same as above
                for row in range( self.fragments[0].norbs_frag ):
                    for col in range( row, self.fragments[0].norbs_frag ):
                        H1 = np.zeros( [ self.norbs_tot, self.norbs_tot ], dtype=int )
                        for jumper in range( self.norbs_tot // self.fragments[0].norbs_frag ):
                            jumpsquare = self.fragments[0].norbs_frag * jumper
                            H1[ jumpsquare + row, jumpsquare + col ] = 1
                            H1[ jumpsquare + col, jumpsquare + row ] = 1
                        theH1.append( H1 )
            else: # NO translational invariance assumed
                # same as above
                jumpsquare = 0
                for frag in self.fragments:
                    for row in range( frag.norbs_frag ):
                        for col in range( row, frag.norbs_frag ):
                            H1 = np.zeros( [ self.norbs_tot, self.norbs_tot ], dtype=int )
                            H1[ jumpsquare + row, jumpsquare + col ] = 1
                            H1[ jumpsquare + col, jumpsquare + row ] = 1
                            theH1.append( H1 )
                    jumpsquare += frag.norbs_frag
        return theH1
        
    def doexact( self, chempot_frag=0.0, frag_constrained_casscf=False ):  
        oneRDM_loc = self.helper.construct1RDM_loc( self.doSCF, self.umat ) 
        self.energy = 0.0												
        self.spin = 0.0

        for frag in self.fragments:
            frag.frag_constrained_casscf = frag_constrained_casscf
            frag.solve_impurity_problem (chempot_frag)
            self.energy += frag.E_frag
            self.spin += frag.S2_frag

        
        if (self.doDET and self.doDET_NO):
            self.loc2fno = self.constructloc2fno()

        Nelectrons = sum ((frag.nelec_frag for frag in self.fragments))

        if self.TransInv:
            Nelectrons = Nelectrons * len( self.fragments )
            self.energy = self.energy * len( self.fragments )

		
        # When an incomplete impurity tiling is used for the Hamiltonian, self.energy should be augmented with the remaining HF part
        if ( self.norbs_allcore > 0 ):
        
            if ( self.CC_E_TYPE == 'CASCI' ):
                Nelectrons = np.trace (self.fragments[0].oneRDM_loc) # Because full active space is used to compute the energy
            else:
                #transfo = np.eye( self.norbs_tot, dtype=float )
                #totalOEI  = self.ints.dmet_oei(  transfo, self.norbs_tot )
                #totalFOCK = self.ints.dmet_fock( transfo, self.norbs_tot, oneRDM )
                #self.energy += 0.5 * np.einsum( 'ij,ij->', oneRDM[remainingOrbs==1,:], \
                #         totalOEI[remainingOrbs==1,:] + totalFOCK[remainingOrbs==1,:] )
                #Nelectrons += np.trace( (oneRDM[remainingOrbs==1,:])[:,remainingOrbs==1] )

                assert (np.array_equal(self.ints.active, np.ones([self.ints.mol.nao_nr()], dtype=int)))

                from pyscf import scf
                from types import MethodType
                mol_ = self.ints.mol
                mf_  = scf.RHF(mol_)

                xorb = np.dot(mf_.get_ovlp(), self.ints.ao2loc)
                hc  = -chempot_imp * np.dot(xorb[:,self.is_allcore_orb], xorb[:,self.is_allcore_orb].T)
                dm0 = np.dot(self.ints.ao2loc, np.dot(oneRDM, self.ints.ao2loc.T))

                def mf_hcore (self, mol=None):
                    if mol is None: mol = self.mol
                    return scf.hf.get_hcore(mol) + hc
                mf_.get_hcore = MethodType(mf_hcore, mf_)
                mf_.scf(dm0)
                assert (mf_.converged)

                rdm1 = mf_.make_rdm1()
                jk   = mf_.get_veff(dm=rdm1)

                xorb = np.dot(mf_.get_ovlp(), self.ints.ao2loc)
                rdm1 = np.dot(xorb.T, np.dot(rdm1, xorb))
                oei  = np.dot(self.ints.ao2loc.T, np.dot(mf_.get_hcore()-hc, self.ints.ao2loc))
                jk   = np.dot(self.ints.ao2loc.T, np.dot(jk, self.ints.ao2loc))
                oei_eff = oei + (0.5 * jk)

                AllcoreEnergy = 0.5 * np.einsum('ij,ij->', rdm1[:,is_allcore_orb], oei_eff[:,is_allcore_orb]) \
                              + 0.5 * np.einsum('ij,ij->', rdm1[is_allcore_orb,:], oei_eff[is_allcore_orb,:]) 
                self.energy += ImpEnergy
                Nelectrons += np.trace(rdm1[np.ix_(is_allcore_orb,is_allcore_orb)])
                self.oneRDMallcore_loc = rdm1

        self.energy += self.ints.const()
        print ("Current whole-molecule energy: {0:.6f}".format (self.energy))
        print ("Current whole-molecule spin: {0:.6f}".format (self.spin))
        return Nelectrons
        
    def constructloc2fno( self ):

        myloc2fno = np.zeros ((self.norbs_tot, self.norbs_tot))
        for frag in self.fragments:
            myloc2fno[:,frag.frag_orb_list] = frag.loc2fno
        if self.TransInv:
            raise RuntimeError("fix constructloc2fno before you try to do translationally-invariant NO-basis things")
            norbs_frag = self.fragments[0].norbs_frag
            for it in range( 1, self.norbs_tot / norbs_frag ):
                myloc2fno[ it*norbs_frag:(it+1)*norbs_frag, it*norbs_frag:(it+1)*norbs_frag ] = myloc2fno[ 0:norbs_frag, 0:norbs_frag ]
        '''if True:
            assert ( linalg.norm( np.dot( myloc2fno.T, myloc2fno ) - np.eye( self.umat.shape[0] ) ) < 1e-10 )
            assert ( linalg.norm( np.dot( myloc2fno, myloc2fno.T ) - np.eye( self.umat.shape[0] ) ) < 1e-10 )
        elif (jumpsquare != frag.norbs_tot):
            myloc2fno = mrh.util.basis.complete_a_basis (myloc2fno)'''
        return myloc2fno
        
    def costfunction( self, newumatflat ):

        return linalg.norm( self.rdm_differences( newumatflat ) )**2

    def alt_costfunction( self, newumatflat ):

        newumatsquare_loc = self.flat2square( newumatflat )
        oneRDM_loc = self.helper.construct1RDM_loc( self.doSCF, newumatsquare_loc )

        errors    = self.rdm_differences (numatflat) 
        errors_sq = self.flat2square (errors)

        if self.minFunc == 'OEI' :
            e_fun = np.trace( np.dot(self.ints.loc_oei(), oneRDM_loc) )
        elif self.minFunc == 'FOCK_INIT' :
            e_fun = np.trace( np.dot(self.ints.loc_rhf_fock(), oneRDM_loc) )
        # e_cstr = np.sum( newumatflat * errors )    # not correct, but gives correct verify_gradient results
        e_cstr = np.sum( newumatsquare_loc * errors_sq )
        return -e_fun-e_cstr
        
    def costfunction_derivative( self, newumatflat ):
        
        errors = self.rdm_differences( newumatflat )
#        error_derivs = self.rdm_differences_derivative( newumatflat )
        thegradient = np.zeros([ len( newumatflat ) ])
#        for counter in range( len( newumatflat ) ):
        idx = 0
        for error_derivs in self.rdm_differences_derivative (newumatflat):
            #thegradient[ counter ] = 2 * np.sum( np.multiply( error_derivs[ : , counter ], errors ) )
            #thegradient[ counter ] = 2 * np.dot( error_derivs[ : , counter ], errors )
            thegradient[ idx ] = 2 * np.dot( error_derivs, errors )
            idx += 1
        assert (idx == len (newumatflat))
        return thegradient

    def alt_costfunction_derivative( self, newumatflat ):
        
#        errors = self.rdm_differences_bis( newumatflat )
        errors = self.rdm_differences (numatflat) # I should have collapsed the function of rdm_differences_bis into rdm_differences
        return -errors

    def rdm_differences( self, newumatflat ):
    
        self.acceptable_errvec_check ()
        newumatsquare_loc = self.flat2square( newumatflat )

        oneRDM_loc = self.helper.construct1RDM_loc( self.doSCF, newumatsquare_loc )
        errvec = np.concatenate ([frag.get_errvec (self, oneRDM_loc) for frag in self.fragments])
        
        return errvec

    def rdm_differences_derivative( self, newumatflat ):
        
        self.acceptable_errvec_check ()
        newumatsquare_loc = self.flat2square( newumatflat )
        # RDMderivs_rot appears to be in the natural-orbital basis if doDET_NO is specified and the local basis otherwise
        RDMderivs_rot = self.helper.construct1RDM_response( self.doSCF, newumatsquare_loc, self.loc2fno )
        gradient = []
        for countgr in range( len( newumatflat ) ):
            # The projection below should do nothing for ordinary DMET, but when a wma space is used it should prevent the derivative from pointing into the wma space
            rsp_1RDM = RDMderivs_rot[countgr,:,:]
            if self.mc_dmet:
                rsp_1RDM = project_operator_into_subspace (RDMderivs_rot[countgr, :, :], self.ints.loc2idem) 
            errvec = np.concatenate ([frag.get_rsp_1RDM_elements (self, rsp_1RDM) for frag in self.fragments])
            yield errvec
#            gradient.append( errvec )
#        gradient = np.array( gradient ).T
        
#        return gradient
        
    def verify_gradient( self, umatflat ):
    
        gradient = self.costfunction_derivative( umatflat )
        cost_reference = self.costfunction( umatflat )
        gradientbis = np.zeros( [ len( gradient ) ])
        stepsize = 1e-7
        for cnt in range( len( gradient ) ):
            umatbis = np.array( umatflat, copy=True )
            umatbis[cnt] += stepsize
            costbis = self.costfunction( umatbis )
            gradientbis[ cnt ] = ( costbis - cost_reference ) / stepsize
        print ("   Norm( gradient difference ) =", linalg.norm( gradient - gradientbis ))
        print ("   Norm( gradient )            =", linalg.norm( gradient ))
        
    def hessian_eigenvalues( self, umatflat ):
    
        stepsize = 1e-7
        print ('Calculating hessian eigenvalues...')
        grad_start = time.time ()
        gradient_reference = self.costfunction_derivative( umatflat )
        grad_end = time.time ()
        print ("Gradient-reference calculated in {} seconds".format (grad_end - grad_start))
        print ("Gradient is an array of length {}".format (gradient_reference.shape))
        print ("Hessian is a {}-by-{} matrix".format (len (umatflat), len (umatflat)))
        hess_start = time.time ()
        hessian = np.zeros( [ len( umatflat ), len( umatflat ) ] )
        for cnt in range( len( umatflat ) ):
            gradient = umatflat.copy()
            gradient[ cnt ] += stepsize
            gradient = self.costfunction_derivative( gradient )
            hessian[ :, cnt ] = ( gradient - gradient_reference ) / stepsize
        hessian = 0.5 * ( hessian + hessian.T )
        hess_end = time.time ()
        print ("Hessian evaluated in {} seconds".format (hess_end - hess_start))
        diag_start = time.time ()
        eigvals, eigvecs = linalg.eigh( hessian )
        diag_end = time.time ()
        idx = eigvals.argsort()
        eigvals = eigvals[ idx ]
        eigvecs = eigvecs[ :, idx ]
        print ("Hessian eigenvalues =", eigvals)
        #print "Hessian 1st eigenvector =",eigvecs[:,0]
        #print "Hessian 2nd eigenvector =",eigvecs[:,1]
        
    def flat2square( self, umatflat ):
    
        umatsquare = np.zeros( [ self.norbs_tot, self.norbs_tot ], )
        umat_idx = np.diag_indices (self.norbs_tot) if self.doDET else self.umat_ftriu_idx
        umatsquare[ umat_idx ] = umatflat
        umatsquare = umatsquare.T
        umatsquare[ umat_idx ] = umatflat
        if ( self.TransInv == True ):
            assert (False), "No translational invariance until you fix it!"
            norbs_frag = self.fragments[0].norbs_frag
            for it in range( 1, self.norbs_tot // norbs_frag ):
                umatsquare[ it*norbs_frag:(it+1)*norbs_frag, it*norbs_frag:(it+1)*norbs_frag ] = umatsquare[ 0:norbs_frag, 0:norbs_frag ]
                
        '''if True:
            umatsquare_bis = np.zeros( [ self.norbs_tot, self.norbs_tot ], dtype=float )
            for cnt in range( len( umatflat ) ):
                umatsquare_bis += umatflat[ cnt ] * self.helper.list_H1[ cnt ]
            print "Verification flat2square = ", linalg.norm( umatsquare - umatsquare_bis )'''
        
        if ( self.loc2fno != None ):
            umatsquare = np.dot( np.dot( self.loc2fno, umatsquare ), self.loc2fno.T )
        return umatsquare
        
    def square2flat( self, umatsquare ):
    
        umatsquare_bis = np.array( umatsquare, copy=True )
        if ( self.loc2fno != None ):
            umatsquare_bis = np.dot( np.dot( self.loc2fno.T, umatsquare_bis ), self.loc2fno )
        umat_idx = np.diag_indices (self.norbs_tot) if self.doDET else self.umat_ftriu_idx
        umatflat = umatsquare_bis[ umat_idx ]
        return umatflat
        
    def numeleccostfunction( self, chempot_imp ):
        
        Nelec_dmet   = self.doexact (chempot_imp)
        Nelec_target = self.ints.nelec_tot
        print ("      (chemical potential , number of electrons) = (", chempot_imp, "," , Nelec_dmet ,")")
        return Nelec_dmet - Nelec_target

    def doselfconsistent (self):
    
        #scfinit = tracemalloc.take_snapshot ()
        #scfinit.dump ('scfinit.snpsht')
        iteration = 0
        u_diff = 1.0
        self.mu_imp = self.chempot_init
        convergence_threshold = 1e-6
        rdm = np.zeros ((self.norbs_tot, self.norbs_tot))
        print ("RHF energy =", self.ints.fullEhf)

        while (u_diff > convergence_threshold):
            u_diff, rdm = self.doselfconsistent_corrpot (rdm, [('corrpot', iteration)])
            iteration += 1 
            if iteration > self.corrpot_maxiter:
                raise RuntimeError ('Maximum correlation-potential cycles!')

        if ( self.CC_E_TYPE == 'CASCI' ):
            assert( len (self.fragments) == 1 )		
            print("-----NOTE: CASCI or Single embedding is used-----")				
            self.energy = self.fragments[0].E_imp
        if self.debug_energy:
            debug_Etot (self)
        print (self.S2_exptval ())

        for frag in self.fragments:
            frag.impurity_molden ('natorb', natorb=True)
            frag.impurity_molden ('imporb')
            frag.impurity_molden ('molorb', molorb=True)
        
        return self.energy

    def doselfconsistent_corrpot (self, rdm_old, iters):
        umat_old = np.array(self.umat, copy=True)
        
        # Find the chemical potential for the correlated impurity problem
        myiter = iters[-1][-1]
        nextiter = 0
        orb_diff = 1.0
        convergence_threshold = 1e-5
        while (orb_diff > convergence_threshold):
            lower_iters = iters + [('orbs', nextiter)]
            orb_diff = self.doselfconsistent_orbs (lower_iters)
            nextiter += 1
            if nextiter > self.orb_maxiter:
                raise RuntimeError ('Maximum active-orbital rotation cycles!')
        #itersnap = tracemalloc.take_snapshot ()
        #itersnap.dump ('iter{}bgn.snpsht'.format (myiter))
        

        # self.verify_gradient( self.square2flat( self.umat ) ) # Only works for self.doSCF == False!!
        # if ( self.SCmethod != 'NONE' and not(self.altcostfunc) ):
        #    self.hessian_eigenvalues( self.square2flat( self.umat ) )
        
        # Solve for the u-matrix
        if ( self.altcostfunc and self.SCmethod == 'BFGS' ):
            result = optimize.minimize( self.alt_costfunction, self.square2flat( self.umat ), jac=self.alt_costfunction_derivative, options={'disp': False} )
            self.umat = self.flat2square( result.x )
        elif ( self.SCmethod == 'LSTSQ' ):
            result = optimize.leastsq( self.rdm_differences, self.square2flat( self.umat ), Dfun=self.rdm_differences_derivative, factor=0.1 )
            self.umat = self.flat2square( result[ 0 ] )
        elif ( self.SCmethod == 'BFGS' ):
            print ("Doing BFGS for chemical potential.....")
            bfgs_start = time.time ()
            result = optimize.minimize( self.costfunction, self.square2flat( self.umat ), jac=self.costfunction_derivative, options={'disp': True} )
            self.umat = self.flat2square( result.x )
            print ("BFGS done after {} seconds".format (time.time () - bfgs_start))
        if self.CC_E_TYPE == 'CASCI':
            # You NEED the diagonal component if the molecule isn't tiled out with fragments!
            # But otherwise, it's a redundant chemical potential term
            self.umat = self.umat - np.eye( self.umat.shape[ 0 ] ) * np.average( np.diag( self.umat ) ) # Remove arbitrary chemical potential shifts
        if ( self.altcostfunc ):
            print ("   Cost function after convergence =", self.alt_costfunction( self.square2flat( self.umat ) ))
        else:
            print ("   Cost function after convergence =", self.costfunction( self.square2flat( self.umat ) ))
        
        # Possibly print the u-matrix / 1-RDM
        if self.print_u:
            self.print_umat()
            np.save ('umat.npy', self.umat)
        if self.print_rdm:
            self.print_1rdm()
        if self.corrpot_mf_molden_cnt < self.corrpot_mf_moldens:
            fname = 'dmet_corrpot.{:d}.molden'.format (self.corrpot_mf_molden_cnt)
            oneRDM_loc = self.helper.construct1RDM_loc( self.doSCF, self.umat )
            mf_occ, loc2mf = matrix_eigen_control_options (oneRDM_loc, sort_vecs=-1)
            mf_coeff = np.dot (self.ints.ao2loc, loc2mf)
            molden.from_mo (self.ints.mol, fname, mf_coeff, occ=mf_occ)
            self.corrpot_mf_molden_cnt += 1
            if self.corrpot_mf_molden_cnt == self.corrpot_mf_moldens:
                self.corrpot_mf_molden_cnt = 0
        
        # Get the error measure
        u_diff = linalg.norm( umat_old - self.umat )
        print ("   2-norm of difference old and new u-mat =", u_diff)
        rdm_new = self.transform_ed_1rdm ()
        rdm_diff = linalg.norm( rdm_old - rdm_new )
        print ("   2-norm of difference old and new 1-RDM =", rdm_diff)
        print ("******************************************************")
        self.umat = self.relaxation * umat_old + ( 1.0 - self.relaxation ) * self.umat

        u_diff = rdm_diff        
        if ( self.SCmethod == 'NONE' ):
            u_diff = 0 # Do only 1 iteration

        #itersnap = tracemalloc.take_snapshot ()
        #itersnap.dump ('iter{}end.snpsht'.format (myiter))

        return u_diff, rdm_new

    def doselfconsistent_orbs (self, iters):

        loc2wmas_old = np.concatenate ([frag.loc2amo for frag in self.fragments], axis=1)
        loc2wmcs_old = get_complementary_states (loc2wmas_old)
        mc_dmet_switch = int (self.mc_dmet)
        if self.do_refragmentation and self.mc_dmet:
            mc_dmet_switch = 2
        if self.mc_dmet:
            self.ints.setup_wm_core_scf (self.fragments)

        oneRDM_loc = self.helper.construct1RDM_loc( self.doSCF, self.umat )
        if self.do_refragmentation and self.mc_dmet:
            oneRDM_loc = self.refragmentation (loc2wmas_old, loc2wmcs_old, oneRDM_loc)
        else:
            for frag in self.fragments:
                frag.restore_default_embedding_basis ()

        self.energy = 0.0
        self.spin = 0.0
        for frag in self.fragments:
            frag.do_Schmidt (oneRDM_loc, self.fragments, loc2wmcs_old, mc_dmet_switch)
            frag.construct_impurity_hamiltonian ()
        if self.examine_ifrag_olap:
            examine_ifrag_olap (self)
        if self.examine_wmcs:
            examine_wmcs (self)

        for itertype, iteridx in iters:
            print ("{0} iteration {1}".format (itertype, iteridx))

        # Iterate on chemical potential
        if self.CC_E_TYPE == 'CASCI' or self.noselfconsistent:
            self.mu_imp = 0.0
            self.doexact (self.mu_imp)
        else:
            self.mu_imp = optimize.newton( self.numeleccostfunction, self.mu_imp, tol=self.chempot_tol )
            print ("   Chemical potential =", self.mu_imp)
        for frag in self.fragments:
            frag.impurity_molden ('natorb', natorb=True)
            frag.impurity_molden ('imporb')
            frag.impurity_molden ('molorb', molorb=True)
        self.doexact (self.mu_imp, frag_constrained_casscf=self.do_constrCASSCF)
        #if self.mc_dmet:
        #    self.ints.setup_wm_core_scf (self.fragments)
        
        loc2wmas_new = np.concatenate ([frag.loc2amo for frag in self.fragments], axis=1)
        try:
            orb_diff = measure_basis_olap (loc2wmas_new, loc2wmcs_old)[0] / max (1,loc2wmas_new.shape[1])
        except:
            raise RuntimeError("what?\n{0}\n{1}".format(loc2wmas_new.shape,loc2wmcs_old.shape))

        print ("Whole-molecule active-space orbital shift = {0}".format (orb_diff))
        if self.mc_dmet == False:
            orb_diff = 0 # Do only 1 iteration

        return orb_diff 
        
    def print_umat( self ):
    
        print ("The u-matrix =")
        squarejumper = 0
        for frag in self.fragments:
            print (self.umat[ squarejumper:squarejumper+frag.norbs_frag , squarejumper:squarejumper+frag.norbs_frag ])
            squarejumper += frag.norbs_frag
    
    def print_1rdm( self ):
    
        print ("The ED 1-RDM of the impurities ( + baths ) =")
        for frag in self.fragments:
            print (frag.get_oneRDM_imp ())
            
    def transform_ed_1rdm( self ):
    
        norbs_frag = [f.loc2frag.shape[1] for f in self.fragments]
        result = np.zeros( [sum (norbs_frag), sum (norbs_frag)], dtype = np.float64)
        loc2frag = np.concatenate ([f.loc2frag for f in self.fragments], axis=1)
        assert (is_basis_orthonormal (loc2frag))
        assert (sum (norbs_frag) == loc2frag.shape[1])
        frag_ranges = [sum (norbs_frag[:i]) for i in range (len (norbs_frag) + 1)]
        for idx, frag in enumerate (self.fragments):
            i = frag_ranges[idx]
            j = frag_ranges[idx+1]
            result[i:j,i:j] = frag.get_oneRDM_frag ()
        return represent_operator_in_basis (result, loc2frag.conjugate ().T)
        
    def dump_bath_orbs( self, filename, frag_idx=0 ):
        
        from mrh.my_dmet import qcdmet_paths
        from pyscf import tools
        from pyscf.tools import molden
        with open( filename, 'w' ) as thefile:
            molden.header( self.ints.mol, thefile )
            molden.orbital_coeff( self.ints.mol, thefile, np.dot( self.ints.ao2loc, self.fragment[frag_idx].loc2imp ) )
    
    def onedm_solution_rhf(self):
        return self.helper.construct1RDM_loc( self.doSCF, self.umat )
   
    def refragmentation (self, loc2wmas, loc2wmcs, oneRDM_loc):
        ''' Separately ortho-localize the whole-molecule active orbitals and whole-molecule core orbitals and adjust the RDM and CDMs. '''

        # Don't waste time if there are no active orbitals
        if loc2wmas.shape[1] == 0:
            return oneRDM_loc

        # Examine the states that turn into I relocalize if the debug is specified.
        if self.debug_reloc:
            ao2before = np.dot (self.ints.ao2loc, np.append (loc2wmas, loc2wmcs, axis=1))

        # Separately localize and assigne the inactive and active orbitals as fragment orbitals
        loc2wmas = self.ints.relocalize_states (orthonormalize_a_basis (loc2wmas), self.fragments, oneRDM_loc, natorb=True)
        loc2wmcs = self.ints.relocalize_states (loc2wmcs, self.fragments, oneRDM_loc, canonicalize=True)

        # Evaluate how many electrons are in each active subspace
        for loc2amo, frag in zip (loc2wmas, self.fragments):
            nelec_amo = np.einsum ("ip,ij,jp->",loc2amo.conjugate (), oneRDM_loc, loc2amo)
            interr = nelec_amo - round (nelec_amo)
            print ("{} fragment has {:.5f} active electrons (integer error: {:.2e})".format (frag.frag_name, nelec_amo, interr))
            #assert (interr < self.nelec_int_thresh), "Fragment with non-integer number of electrons appears"
            interr = np.eye (loc2amo.shape[1]) * interr / loc2amo.shape[1]
            oneRDM_loc -= reduce (np.dot, [loc2amo, interr, loc2amo.conjugate ().T])
            
        # Evaluate the entanglement of the active subspaces
        for (o1, f1), (o2, f2) in combinations (zip (loc2wmas, self.fragments), 2):
            if o1.shape[1] > 0 and o2.shape[1] > 0:
                RDM_12 = reduce (np.dot, [o1.conjugate ().T, oneRDM_loc, o2])
                RDM_12_sq = np.dot (RDM_12.conjugate ().T, RDM_12)
                evals, evecs = linalg.eigh (RDM_12_sq)
                entanglement = sum (evals)
                print ("Fragments {} and {} have an active-space entanglement trace of {:.2e}".format (f1.frag_name, f2.frag_name, entanglement))

        # Examine localized states if necessary
        # Put this at the end so if I get all the way to another orbiter and assert-fails on the integer thing, I can look at the ~last~ set of relocalized orbitals
        if self.debug_reloc:
            fock_loc = self.ints.loc_rhf_fock_bis (oneRDM_loc)
            loc2c = np.concatenate (loc2wmcs, axis=1)
            loc2a = np.concatenate (loc2wmas, axis=1)
            lo_ene, evecs_c = matrix_eigen_control_options (represent_operator_in_basis (fock_loc, loc2c), sort_vecs=1, only_nonzero_vals=False)
            loc2c = np.dot (loc2c, evecs_c)
            lo_ene = np.append (-999 * np.ones (loc2a.shape[1]), lo_ene)
            loc2after = np.append (loc2a, loc2c, axis=1)
            lo_occ = np.einsum ('ip,ij,jp->p', loc2after, oneRDM_loc, loc2after)
            ao2after = np.dot (self.ints.ao2loc, loc2after)
            occ = np.zeros (ao2before.shape[1])
            occ[:sum([l.shape[1] for l in loc2wmas])] = 1
            molden.from_mo (self.ints.mol, "main_object.refragmentation.before.molden", ao2before, occ=occ)
            molden.from_mo (self.ints.mol, "main_object.refragmentation.after.molden", ao2after, occ=lo_occ, ene=lo_ene)

        for loc2imo, loc2amo, frag in zip (loc2wmcs, loc2wmas, self.fragments):
            frag.set_new_fragment_basis (np.append (loc2imo, loc2amo, axis=1))

        return oneRDM_loc

    def S2_exptval (self, fragments = None):

        if fragments is None:
            fragments = self.fragments

        S2_tot = 0
        print ("Spin expectation value fragment breakdown:")
        for f1, f2 in product (fragments, repeat=2):
            if f1 is f2:
                oneRDM = f1.get_oneRDM_frag ()
                twoCDM = represent_operator_in_basis (f1.twoCDM_imp, f1.imp2frag)
                S2 = S2_exptval (oneRDM, twoCDM, cumulant=True)
            else:
                DM_pq = (represent_operator_in_basis (f1.oneRDM_loc, f1.loc2frag, f2.loc2frag)
                       + represent_operator_in_basis (f2.oneRDM_loc, f1.loc2frag, f2.loc2frag)) / 2
                DM_qp = (represent_operator_in_basis (f1.oneRDM_loc, f2.loc2frag, f1.loc2frag)
                       + represent_operator_in_basis (f2.oneRDM_loc, f2.loc2frag, f1.loc2frag)) / 2
                S2 = -3 * np.einsum ('pq,qp->', DM_pq, DM_qp) / 8
                P1 = np.dot (f1.imp2frag, f1.frag2imp)
                P2 = reduce (np.dot, (f1.imp2loc, f2.loc2frag, f2.frag2loc, f1.loc2imp))
                S2 -= np.einsum ('ad,da->', np.einsum ('abcd,cb->ad', f1.twoCDM_imp, P2), P1) / 4
                S2 -= np.einsum ('ab,ba->', np.einsum ('abcd,cd->ab', f1.twoCDM_imp, P2), P1) / 8
                P1 = reduce (np.dot, (f2.imp2loc, f1.loc2frag, f1.frag2loc, f2.loc2imp))
                P2 = np.dot (f2.imp2frag, f2.frag2imp)
                S2 -= np.einsum ('ad,da->', np.einsum ('abcd,cb->ad', f2.twoCDM_imp, P2), P1) / 4
                S2 -= np.einsum ('ab,ba->', np.einsum ('abcd,cd->ab', f2.twoCDM_imp, P2), P1) / 8
            print ("{}-{}: {:.5f}".format (f1.frag_name, f2.frag_name, S2))
            S2_tot += S2
        return S2_tot



