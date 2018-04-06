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

from . import localintegrals, qcdmethelper
import numpy as np
from scipy import optimize
import time

class dmet:

    def __init__( self, theInts, fragments, isTranslationInvariant, SCmethod='LSTSQ', incl_bath_errvec=True, use_constrained_opt=False, doDET=False,doDET_NO=False):
    
        if ( isTranslationInvariant == True ):
            assert( theInts.TI_OK == True )
            assert( len (fragments) == 1 )
        
        assert (( SCmethod == 'LSTSQ' ) or ( SCmethod == 'BFGS' ) or ( SCmethod == 'NONE' ))
        
        self.ints       = theInts
        self.norbs_tot  = self.ints.Norbs
        self.fragments  = fragments
        self.umat       = np.zeros([ self.norbs_tot, self.norbs_tot ], dtype=float)
        self.relaxation = 0.0
        
        self.NI_hack    = False
        self.doSCF      = False
        self.TransInv   = isTranslationInvariant
        self.SCmethod   = SCmethod
        self.CC_E_TYPE  = 'LAMBDA' #'CASCI'/'LAMBDA'
        self.incl_bath_errvec = incl_bath_errvec
        self.doDET      = doDET
        self.doDET_NO   = doDET_NO
        self.loc2fno = None
        self.altcostfunc = use_constrained_opt

        #HP: use UHF solver
        self.UHF = False
		
        self.minFunc    = None
        if self.altcostfunc:
            self.minFunc = 'FOCK_INIT'  # 'OEI'
            assert (self.incl_bath_errvec == False)
            assert (self.doDET == False)
            assert (self.SCmethod == 'BFGS' or self.SCmethod == 'NONE')
       
        if ( self.doDET == True ):
            # Cfr Bulik, PRB 89, 035140 (2014)
            self.incl_bath_errvec = False
        
        self.print_u   = True
        self.print_rdm = True
        
        self.get_allcore_orbs ()
        if ( self.norbs_allcore > 0 ): # One or more impurities which do not cover the entire system
            assert( self.TransInv == False ) # Make sure that you don't work translational invariant
            # Note on working with impurities which do no tile the entire system: they should be the first orbitals in the Hamiltonian!
        
        self.energy   = 0.0
        self.mu_imp   = 0.0
        self.helper   = qcdmethelper.qcdmethelper( self.ints, self.makelist_H1(), self.altcostfunc, self.minFunc )
        
        self.time_ed  = 0.0
        self.time_cf  = 0.0
        self.time_func= 0.0
        self.time_grad= 0.0
        
        np.set_printoptions(precision=3, linewidth=160)
        
    def get_allcore_orbs ( self ):
        # I guess this just determines whether every orbital has a fragment or not    

        quicktest = np.zeros([ self.norbs_tot ], dtype=int)
        for frag in self.fragments:
            quicktest += np.abs(frag.is_frag_orb.astype (int))
        assert( np.all( quicktest >= 0 ) )
        assert( np.all( quicktest <= 1 ) )
        self.is_allcore_orb = np.logical_not (quicktest.astype (bool))
            
    @property
    def loc2allcore (self):
        return np.eye (self.norbs_tot, dtype=float)[:,self.is_allcore_orb]

    @property
    def norbs_allcore (self):
        return np.count_nonzero (self.is_allcore_orb)

    @property
    def allcore_orb_list (self):
        return np.flatnonzero (self.is_allcore_orb)

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
        
    def doexact( self, chempot_frag=0.0 ):  
        OneRDM = self.helper.construct1RDM_loc( self.doSCF, self.umat ) #HP: construct 1rdm (in AO) for the total system from the Fock matrix in local ao basis. projected on atoms?
        self.energy   = 0.0												#This OneRDM will be used to construct the bath

        for frag in self.fragments:
            frag.do_Schmidt_decomposition (OneRDM)
            frag.construct_impurity_hamiltonian (OneRDM)
            frag.solve_impurity_problem (chempot_frag, self.CC_E_TYPE)
            self.energy += frag.E_frag
        
        if ( self.doDET == True ) and ( self.doDET_NO == True ):
            self.loc2fno = self.constructloc2fno()
        
        Nefrag = [np.trace (frag.oneRDM_imp[:frag.norbs_frag,:frag.norbs_frag]) for frag in self.fragments]
        Nelectrons = sum (Nefrag)
			
        if ( self.TransInv == True ):
            Nelectrons = Nelectrons * len( self.fragments )
            self.energy = self.energy * len( self.fragments )

		#HungPham		
        print('Fragment energies:', [frag.E_frag for frag in self.fragments])
        print('Fragment electrons:',Nefrag)
        E1 = self.energy	
        print('DEBUG Energy before adding up envi contribution:',E1)
		
        # When an incomplete impurity tiling is used for the Hamiltonian, self.energy should be augmented with the remaining HF part
        if ( self.norbs_allcore > 0 ):
        
            if ( self.CC_E_TYPE == 'CASCI' ):
                assert( len (self.fragments) == 1 )		
                print("-----NOTE: CASCI or Single embedding is used-----")				
                self.energy = self.frag[0].E_imp
                Nelectrons = np.trace( self.frag[0].oneRDM_loc ) # Because full active space is used to compute the energy
            else:
                #transfo = np.eye( self.norbs_tot, dtype=float )
                #totalOEI  = self.ints.dmet_oei(  transfo, self.norbs_tot )
                #totalFOCK = self.ints.dmet_fock( transfo, self.norbs_tot, OneRDM )
                #self.energy += 0.5 * np.einsum( 'ij,ij->', OneRDM[remainingOrbs==1,:], \
                #         totalOEI[remainingOrbs==1,:] + totalFOCK[remainingOrbs==1,:] )
                #Nelectrons += np.trace( (OneRDM[remainingOrbs==1,:])[:,remainingOrbs==1] )

                assert (np.array_equal(self.ints.active, np.ones([self.ints.mol.nao_nr()], dtype=int)))

                from pyscf import scf
                from types import MethodType
                mol_ = self.ints.mol
                mf_  = scf.RHF(mol_)

                xorb = np.dot(mf_.get_ovlp(), self.ints.ao2loc)
                hc  = -chempot_imp * np.dot(xorb[:,self.is_allcore_orb], xorb[:,self.is_allcore_orb].T)
                dm0 = np.dot(self.ints.ao2loc, np.dot(OneRDM, self.ints.ao2loc.T))

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

                ImpEnergy = \
                   + 0.50 * np.einsum('ji,ij->', rdm1[:,is_allcore_orb], oei[impOrbs,:]) \
                   + 0.50 * np.einsum('ji,ij->', rdm1[is_allcore_orb,:], oei[:,impOrbs]) \
                   + 0.25 * np.einsum('ji,ij->', rdm1[:,is_allcore_orb], jk[impOrbs,:]) \
                   + 0.25 * np.einsum('ji,ij->', rdm1[is_allcore_orb,:], jk[:,impOrbs])
                self.energy += ImpEnergy
                Nelectrons += np.trace(rdm1[np.ix_(is_allcore_orb,impOrbs)])

        print('Energy decomposition for debug:',E1, self.energy-E1, self.energy) #HP: for debug
        print('Nuclear potential:',self.ints.const()) 		
        self.energy += self.ints.const()
        return Nelectrons
        
    def constructloc2fno( self ):
    
        myloc2fno = np.empty ((0,0), dtype=float )
        jumpsquare = 0
        for frag in self.fragments:
            myloc2fno = np.append (myloc2fno, frag.loc2fno)
            jumpsquare += frag.norbs_frag
        if ( self.TransInv == True ):
            norbs_frag = self.fragments[0].norbs_frag
            for it in range( 1, self.norbs_tot / norbs_frag ):
                myloc2fno[ it*norbs_frag:(it+1)*norbs_frag, it*norbs_frag:(it+1)*norbs_frag ] = myloc2fno[ 0:norbs_frag, 0:norbs_frag ]
        '''if True:
            assert ( np.linalg.norm( np.dot( myloc2fno.T, myloc2fno ) - np.eye( self.umat.shape[0] ) ) < 1e-10 )
            assert ( np.linalg.norm( np.dot( myloc2fno, myloc2fno.T ) - np.eye( self.umat.shape[0] ) ) < 1e-10 )'''
        elif (jumpsquare != frag.norbs_tot):
            myloc2fno = mrh.util.basis.complete_a_basis (myloc2fno)
        return myloc2fno
        
    def costfunction( self, newumatflat ):

        return np.linalg.norm( self.rdm_differences( newumatflat ) )**2

    def alt_costfunction( self, newumatflat ):

        newumatsquare_loc = self.flat2square( newumatflat )
        OneRDM_loc = self.helper.construct1RDM_loc( self.doSCF, newumatsquare_loc )

#        errors    = self.rdm_differences_bis( newumatflat )
        errors    = self.rdm_differences (numatflat) # I should have collapsed the function of rdm_differences_bis into rdm_differences
        errors_sq = self.flat2square (errors)

        if self.minFunc == 'OEI' :
            e_fun = np.trace( np.dot(self.ints.loc_oei(), OneRDM_loc) )
        elif self.minFunc == 'FOCK_INIT' :
            e_fun = np.trace( np.dot(self.ints.loc_rhf_fock(), OneRDM_loc) )
        # e_cstr = np.sum( newumatflat * errors )    # not correct, but gives correct verify_gradient results
        e_cstr = np.sum( newumatsquare_loc * errors_sq )
        return -e_fun-e_cstr
        
    def costfunction_derivative( self, newumatflat ):
        
        errors = self.rdm_differences( newumatflat )
        error_derivs = self.rdm_differences_derivative( newumatflat )
        thegradient = np.zeros([ len( newumatflat ) ], dtype=float)
        for counter in range( len( newumatflat ) ):
            thegradient[ counter ] = 2 * np.sum( np.multiply( error_derivs[ : , counter ], errors ) )
        return thegradient

    def alt_costfunction_derivative( self, newumatflat ):
        
#        errors = self.rdm_differences_bis( newumatflat )
        errors = self.rdm_differences (numatflat) # I should have collapsed the function of rdm_differences_bis into rdm_differences
        return -errors

    def len_errvec (self):
        if self.doDET:
            # Diagonal elements only
            assert (not self.incl_bath_errvec)
            return np.sum ([frag.norbs_frag for frag in self.fragments])
        elif self.altcostfunc:
            # Upper triangular elements only
            assert (not self.incl_bath_errvec)
            return np.sum ([(frag.norbs_frag ** 2 + frag.norbs_frag) // 2 for frag in self.fragments])
        elif self.incl_bath_errvec:
            # Fragment and bath orbital diagonals and off-diagonals
            return np.sum ([frag.norbs_imp ** 2 for frag in self.fragments])
        else:
            # Fragment-orbital only diagonals and off-diagonals
            return np.sum ([frag.norbs_frag ** 2 for frag in self.fragments])

    def unroll_err1RDM (self, frag, err_1RDM_imp):
        # Fragment and bath diagonals and off-diagonals
        if self.incl_bath_errvec:
            return err_1RDM_imp.flatten (order='F')
        err_1RDM_frag = err_1RDM_imp[:frag.norbs_frag,:frag.norbs_frag]
        if self.doDET:
            if self.doDET_NO:
        # Fragment natural-orbital diagonals
                return np.diag (np.dot (np.dot (frag.frag2fno.T, err_1RDM_frag), frag.frag2fno))
        # Fragment local diagonals
            return np.diag (err_1RDM_frag)
        # Upper-triangular fragment diagonals and off-diagonals
        if self.altcostfunc:
            return err_1RDM_frag[np.triu_indices (frag.norbs_frag)]
        # All fragment diagonals and off-diagonals
        return err_1RDM_frag.flatten (order='F')

    def rdm_differences( self, newumatflat ):
    
        start_func = time.time()
    
        newumatsquare_loc = self.flat2square( newumatflat )
        OneRDM_loc = self.helper.construct1RDM_loc( self.doSCF, newumatsquare_loc )
        errvec = np.empty (0, dtype=float)
        
        for frag in self.fragments:
            mf_1RDM = np.dot (np.dot (frag.loc2imp.T, OneRDM_loc), frag.loc2imp)
            ed_1RDM = frag.oneRDM_imp
            error_1RDM = mf_1RDM - ed_1RDM
            errvec = np.append (errvec, self.unroll_err1RDM (frag, err_1RDM_imp))
        assert (len (errvec.shape) == 1), "{0}".format (errvec.shape)
        assert (errvec.shape[0] = self.len_errvec ()), "{0}".format (errvec.shape[0])

        stop_func = time.time()
        self.time_func += ( stop_func - start_func )
        
        return errvec

'''
    I think I collapsed the function of this into rdm_differences when I defined unroll_err1RDM 
    def rdm_differences_bis( self, newumatflat ):
    
        start_func = time.time()
    
        newumatsquare_loc = self.flat2square( newumatflat )
        OneRDM_loc = self.helper.construct1RDM_loc( self.doSCF, newumatsquare_loc )
        errvec = np.empty (0, dtype=float)

        for frag in self.fragments:
            mf_1RDM = 
#        for count in range( len( self.imp_size ) ): # self.imp_size has length 1 if self.TransInv
            mf_1RDM = OneRDM_loc[np.ix_(frag.frag_orb_list, frag.frag_orb_list)]
            ed_1RDM = frag.oneRDM_imp[:frag.norbs_frag,:frag.norbs_frag]
            theerror = mf_1RDM - ed_1RDM
            # squaresize = theerror.shape[0] * theerror.shape[1]
            # errors[ jump : jump + squaresize ] = np.reshape( theerror, squaresize, order='F' )
            mask_t = self.mask[ np.ix_(range(jumpc,jumpc+self.imp_size[count]),range(jumpc,jumpc+self.imp_size[count])) ]
            squaresize = np.count_nonzero( mask_t )
            errors[ jump : jump + squaresize ] = np.reshape( theerror[mask_t], squaresize, order='F' )
            jump  += squaresize
            jumpc += self.imp_size[count]
        assert ( jump == thesize )
        
        stop_func = time.time()
        self.time_func += ( stop_func - start_func )
        
        return errors
'''

    def unroll_deriv1RDM (self, frag, deriv1RDM_loc):
        # altcostfunc doesn't appear here because its derivative is computed differently
        if self.incl_bath_errvec:
            return np.dot (np.dot (frag.loc2imp.T, deriv1RDM_loc), frag.loc2imp).flatten (order='F')
        deriv1RDM_frag = np.dot (np.dot (frag.loc2frag.T, deriv1RDM_loc), frag.loc2frag)
        if self.doDET_NO:
            deriv1RDM_fno = np.dot (np.dot (frag.frag2fno.T, deriv1RDM_frag), frag.frag2fno)
            return np.diag (deriv1RDM_fno)
        if self.doDET:
            return np.diag (deriv1RDM_frag)
        return deriv1RDM_frag.flatten ()

    def rdm_differences_derivative( self, newumatflat ):
        
        start_grad = time.time()
        
        newumatsquare_loc = self.flat2square( newumatflat )
        # RDMderivs_rot appears to be in the natural-orbital basis if doDET_NO is specified and the local basis otherwise
        RDMderivs_rot = self.helper.construct1RDM_response( self.doSCF, newumatsquare_loc, self.loc2fno )
        
        gradient = []
        for countgr in range( len( newumatflat ) ):
            errvec = np.empty (0, dtype=float )
            deriv1RDM_loc = RDMderivs_rot[countgr, :, :]
            if doDET_NO:
                # It comes out of construct1RDM in the NO basis; I need to change it BACK to make sense of anything
                deriv1RDM_loc = np.dot (np.dot (self.loc2fno, deriv1RDM_loc), self.loc2fno.T)
            for frag in self.fragments:
                errvec = np.append (errvec, self.unroll_deriv1RDM (frag, deriv1RDM_loc))
                assert (len (errvec.shape) == 1), "{0}".format (errvec.shape)
                assert (errvec.shape[0] = self.len_errvec ()), "{0}".format (errvec.shape[0])
            gradient.append( error_deriv )
        gradient = np.array( gradient ).T
        
        stop_grad = time.time()
        self.time_grad += ( stop_grad - start_grad )
        
        return gradient
        
    def verify_gradient( self, umatflat ):
    
        gradient = self.costfunction_derivative( umatflat )
        cost_reference = self.costfunction( umatflat )
        gradientbis = np.zeros( [ len( gradient ) ], dtype=float )
        stepsize = 1e-7
        for cnt in range( len( gradient ) ):
            umatbis = np.array( umatflat, copy=True )
            umatbis[cnt] += stepsize
            costbis = self.costfunction( umatbis )
            gradientbis[ cnt ] = ( costbis - cost_reference ) / stepsize
        print ("   Norm( gradient difference ) =", np.linalg.norm( gradient - gradientbis ))
        print ("   Norm( gradient )            =", np.linalg.norm( gradient ))
        
    def hessian_eigenvalues( self, umatflat ):
    
        stepsize = 1e-7
        gradient_reference = self.costfunction_derivative( umatflat )
        hessian = np.zeros( [ len( umatflat ), len( umatflat ) ], dtype=float )
        for cnt in range( len( umatflat ) ):
            gradient = umatflat.copy()
            gradient[ cnt ] += stepsize
            gradient = self.costfunction_derivative( gradient )
            hessian[ :, cnt ] = ( gradient - gradient_reference ) / stepsize
        hessian = 0.5 * ( hessian + hessian.T )
        eigvals, eigvecs = np.linalg.eigh( hessian )
        idx = eigvals.argsort()
        eigvals = eigvals[ idx ]
        eigvecs = eigvecs[ :, idx ]
        print ("Hessian eigenvalues =", eigvals)
        #print "Hessian 1st eigenvector =",eigvecs[:,0]
        #print "Hessian 2nd eigenvector =",eigvecs[:,1]
        
    def flat2square( self, umatflat ):
    
        umatsquare = np.zeros( [ self.norbs_tot, self.norbs_tot ], dtype=float )
        umatsquare[ np.triu_indices (self.norbs_tot) ] = umatflat
        umatsquare = umatsquare.T
        umatsquare[ np.triu_indices (self.norbs_tot) ] = umatflat
        if ( self.TransInv == True ):
            norbs_frag = self.fragments[0].norbs_frag
            for it in range( 1, self.norbs_tot // norbs_frag ):
                umatsquare[ it*norbs_frag:(it+1)*norbs_frag, it*norbs_frag:(it+1)*norbs_frag ] = umatsquare[ 0:norbs_frag, 0:norbs_frag ]
                
        '''if True:
            umatsquare_bis = np.zeros( [ self.norbs_tot, self.norbs_tot ], dtype=float )
            for cnt in range( len( umatflat ) ):
                umatsquare_bis += umatflat[ cnt ] * self.helper.list_H1[ cnt ]
            print "Verification flat2square = ", np.linalg.norm( umatsquare - umatsquare_bis )'''
        
        if ( self.loc2fno != None ):
            umatsquare = np.dot( np.dot( self.loc2fno, umatsquare ), self.loc2fno.T )
        return umatsquare
        
    def square2flat( self, umatsquare ):
    
        umatsquare_bis = np.array( umatsquare, copy=True )
        if ( self.loc2fno != None ):
            umatsquare_bis = np.dot( np.dot( self.loc2fno.T, umatsquare_bis ), self.loc2fno )
        umatflat = umatsquare_bis[ np.triu_indices (self.norbs_tot) ]
        return umatflat
        
    def numeleccostfunction( self, chempot_imp ):
        
        Nelec_dmet   = self.doexact( chempot_imp )
        Nelec_target = self.ints.Nelec
        print ("      (chemical potential , number of electrons) = (", chempot_imp, "," , Nelec_dmet ,")")
        return Nelec_dmet - Nelec_target

    def doselfconsistent( self ):
    
        iteration = 0
        u_diff = 1.0
        convergence_threshold = 1e-5
        print ("RHF energy =", self.ints.fullEhf)
        
        while ( u_diff > convergence_threshold ):
        
            iteration += 1
            print ("DMET iteration", iteration)
            umat_old = np.array( self.umat, copy=True )
            rdm_old = self.transform_ed_1rdm() # At the very first iteration, this matrix will be zero
            
            # Find the chemical potential for the correlated impurity problem
            start_ed = time.time()
            if (( self.method == 'CC' ) and ( self.CC_E_TYPE == 'CASCI' )):
                self.mu_imp = 0.0
                self.doexact( self.mu_imp )
            elif (( self.method == 'CASSCF' ) and ( self.CC_E_TYPE == 'CASCI' )):
                self.mu_imp = 0.0
                self.doexact( self.mu_imp )
            elif (( self.method == 'RHF' ) and ( self.CC_E_TYPE == 'CASCI' )):
                self.mu_imp = 0.0
                self.doexact( self.mu_imp )				
            else:
                self.mu_imp = optimize.newton( self.numeleccostfunction, self.mu_imp )
                print ("   Chemical potential =", self.mu_imp)
            stop_ed = time.time()
            self.time_ed += ( stop_ed - start_ed )
            print ("   Energy =", self.energy)
            # self.verify_gradient( self.square2flat( self.umat ) ) # Only works for self.doSCF == False!!
            if ( self.SCmethod != 'NONE' and not(self.altcostfunc) ):
                self.hessian_eigenvalues( self.square2flat( self.umat ) )
            
            # Solve for the u-matrix
            start_cf = time.time()
            if ( self.altcostfunc and self.SCmethod == 'BFGS' ):
                result = optimize.minimize( self.alt_costfunction, self.square2flat( self.umat ), jac=self.alt_costfunction_derivative, options={'disp': False} )
                self.umat = self.flat2square( result.x )
            elif ( self.SCmethod == 'LSTSQ' ):
                result = optimize.leastsq( self.rdm_differences, self.square2flat( self.umat ), Dfun=self.rdm_differences_derivative, factor=0.1 )
                self.umat = self.flat2square( result[ 0 ] )
            elif ( self.SCmethod == 'BFGS' ):
                result = optimize.minimize( self.costfunction, self.square2flat( self.umat ), jac=self.costfunction_derivative, options={'disp': False} )
                self.umat = self.flat2square( result.x )
            self.umat = self.umat - np.eye( self.umat.shape[ 0 ] ) * np.average( np.diag( self.umat ) ) # Remove arbitrary chemical potential shifts
            if ( self.altcostfunc ):
                print ("   Cost function after convergence =", self.alt_costfunction( self.square2flat( self.umat ) ))
            else:
                print ("   Cost function after convergence =", self.costfunction( self.square2flat( self.umat ) ))
            stop_cf = time.time()
            self.time_cf += ( stop_cf - start_cf )
            
            # Possibly print the u-matrix / 1-RDM
            if self.print_u:
                self.print_umat()
            if self.print_rdm:
                self.print_1rdm()
            
            # Get the error measure
            u_diff   = np.linalg.norm( umat_old - self.umat )
            rdm_diff = np.linalg.norm( rdm_old - self.transform_ed_1rdm() )
            self.umat = self.relaxation * umat_old + ( 1.0 - self.relaxation ) * self.umat
            print ("   2-norm of difference old and new u-mat =", u_diff)
            print ("   2-norm of difference old and new 1-RDM =", rdm_diff)
            print ("******************************************************")
            
            if ( self.SCmethod == 'NONE' ):
                u_diff = 0.1 * convergence_threshold # Do only 1 iteration
        
        print ("Time cf func =", self.time_func)
        print ("Time cf grad =", self.time_grad)
        print ("Time dmet ed =", self.time_ed)
        print ("Time dmet cf =", self.time_cf)
        
        return self.energy
        
    def print_umat( self ):
    
        print ("The u-matrix =")
        squarejumper = 0
        for frag in self.fragments:
            print (self.umat[ squarejumper:squarejumper+frag.norbs_frag , squarejumper:squarejumper+frag.norbs_frag ])
            squarejumper += frag.norbs_frag
    
    def print_1rdm( self ):
    
        print ("The ED 1-RDM of the impurities ( + baths ) =")
        for frag in self.fragments:
            print (frag.oneRDM_imp)
            
    def transform_ed_1rdm( self ):
    
        result = np.zeros( [self.umat.shape[0], self.umat.shape[0]], dtype=float )
        squarejumper = 0
        for frag in self.fragments:
            result[ squarejumper:squarejumper+frag.norbs_frag , squarejumper:squarejumper+frag.norbs_frag ] = frag.oneRDM_imp[ :frag.norbs_frag , :frag.norbs_frag ]
            squarejumper += frag.norbs_frag
        return result
        
    def dump_bath_orbs( self, filename, frag_idx=0 ):
        
        from . import qcdmet_paths
        from pyscf import tools
        from pyscf.tools import molden
        with open( filename, 'w' ) as thefile:
            molden.header( self.ints.mol, thefile )
            molden.orbital_coeff( self.ints.mol, thefile, np.dot( self.ints.ao2loc, self.fragment[frag_idx].loc2imp ) )
    
    def onedm_solution_rhf(self):
        return self.helper.construct1RDM_loc( self.doSCF, self.umat )
    
