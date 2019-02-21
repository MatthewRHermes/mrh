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
import time, ctypes
#import tracemalloc
from pyscf import scf, mcscf
from pyscf.lo import orth, nao
from pyscf.gto import mole, same_mol
from pyscf.tools import molden
from pyscf.symm.addons import symmetrize_space
from pyscf.scf.addons import project_mo_nr2nr, project_dm_nr2nr
from mrh.util import params
from mrh.util.io import prettyprint_ndarray as prettyprint
from mrh.util.la import matrix_eigen_control_options, matrix_svd_control_options
from mrh.util.basis import represent_operator_in_basis, orthonormalize_a_basis, get_complementary_states, project_operator_into_subspace
from mrh.util.basis import is_matrix_eye, measure_basis_olap, is_basis_orthonormal_and_complete, is_basis_orthonormal, get_overlapping_states
from mrh.util.basis import is_matrix_zero, is_subspace_block_adapted, symmetrize_basis
from mrh.util.rdm import get_2RDM_from_2CDM, get_2CDM_from_2RDM
from mrh.my_dmet.debug import debug_ofc_oneRDM, debug_Etot, examine_ifrag_olap, examine_wmcs
from functools import reduce
from itertools import combinations, product

class dmet:

    def __init__( self, theInts, fragments, calcname='DMET', isTranslationInvariant=False, SCmethod='BFGS', incl_bath_errvec=True, use_constrained_opt=False, 
                    doDET=False, doDET_NO=False, do1SHOT=False, do0SHOT=False, doLASSCF=False, do1EMB=False,
                    minFunc='FOCK_INIT', print_u=True,
                    print_rdm=True, debug_energy=False, debug_reloc=False,
                    nelec_int_thresh=1e-6, chempot_init=0.0, num_mf_stab_checks=0,
                    corrpot_maxiter=50, orb_maxiter=50, chempot_tol=1e-6, corrpot_mf_moldens=0):

        if isTranslationInvariant:
            raise RuntimeError ("The translational invariance option doesn't work!  It needs to be completely rebuilt!")
            assert( theInts.TI_OK == True )
            assert( len (fragments) == 1 )
        
        assert (( SCmethod == 'LSTSQ' ) or ( SCmethod == 'BFGS' ) or ( SCmethod == 'NONE' ))

        #tracemalloc.start (10)

        self.calcname                 = calcname
        self.ints                     = theInts
        self.norbs_tot                = self.ints.norbs_tot
        self.fragments                = fragments
        self.NI_hack                  = False
        self.doSCF                    = False
        self.TransInv                 = isTranslationInvariant
        self.doDET                    = doDET or doDET_NO or do1SHOT or do1EMB
        self.doDET_NO                 = doDET_NO
        self.do1SHOT                  = do1SHOT
        self.do0SHOT                  = do0SHOT
        self.doLASSCF                 = doLASSCF
        self.do1EMB                   = do1EMB
        self.minFunc                  = minFunc
        self.print_u                  = print_u
        self.print_rdm                = print_rdm
        self.SCmethod                 = 'NONE' if (do1SHOT or do0SHOT or doLASSCF or do1EMB) else SCmethod
        self.incl_bath_errvec         = False if self.doDET else incl_bath_errvec
        self.altcostfunc              = False if self.doDET else use_constrained_opt
        self.debug_energy             = debug_energy
        self.debug_reloc              = debug_reloc
        self.nelec_int_thresh         = nelec_int_thresh
        self.chempot                  = chempot_init
        self.num_mf_stab_checks       = num_mf_stab_checks
        self.corrpot_maxiter          = corrpot_maxiter
        self.orb_maxiter              = orb_maxiter
        self.chempot_tol              = chempot_tol
        self.corrpot_mf_moldens       = corrpot_mf_moldens
        self.corrpot_mf_molden_cnt    = 0
        self.ints.num_mf_stab_checks  = num_mf_stab_checks

        for frag in self.fragments:
            frag.debug_energy             = debug_energy
            frag.num_mf_stab_checks       = num_mf_stab_checks
            frag.filehead                 = self.calcname + '_'
            frag.loc2symm                 = self.ints.loc2symm
            frag.ir_names                 = self.ints.mol.irrep_name   
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
        #theH1 = []
        H1start = []
        H1row   = []
        H1col   = []
        H1start.append( 0 )
        totalsize = 0
        def sparsify (list_H1, totalsize, H1start, H1row, H1col):
            rowco, colco = np.where( list_H1 == 1 )
            totalsize += len( rowco )
            H1start.append( totalsize )
            for count2 in range( len( rowco ) ):
                H1row.append( rowco[ count2 ] )
                H1col.append( colco[ count2 ] )
            return totalsize, H1start, H1row, H1col
        if ( self.doDET == True ): # Do density embedding theory
            if ( self.TransInv == True ): # Translational invariance assumed
                # In this case, it appears that H1 identifies a set of diagonal 1RDM elements that are equivalent by symmetry
                for row in range( self.fragments[0].norbs_frag ):
                    H1 = np.zeros( [ self.norbs_tot, self.norbs_tot ], dtype=int )
                    for jumper in range( self.norbs_tot // self.fragments[0].norbs_frag ):
                        jumpsquare = self.fragments[0].norbs_frag * jumper
                        H1[ jumpsquare + row, jumpsquare + row ] = 1
                    #theH1.append( H1 )
                    totalsize, H1start, H1row, H1col = sparsify (H1, totalsize, H1start, H1row, H1col)
            else: # NO translational invariance assumed
                # Huh? In this case it's a long list of giant matrices with only one nonzero value each
                jumpsquare = 0
                for frag in self.fragments:
                    for row in range( frag.norbs_frag ):
                        H1 = np.zeros( [ self.norbs_tot, self.norbs_tot ], dtype=int )
                        H1[ jumpsquare + row, jumpsquare + row ] = 1
                        #theH1.append( H1 )
                        totalsize, H1start, H1row, H1col = sparsify (H1, totalsize, H1start, H1row, H1col)
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
                        #theH1.append( H1 )
                        totalsize, H1start, H1row, H1col = sparsify (H1, totalsize, H1start, H1row, H1col)
            else: # NO translational invariance assumed
                # same as above
                jumpsquare = 0
                for frag in self.fragments:
                    for row in range( frag.norbs_frag ):
                        for col in range( row, frag.norbs_frag ):
                            H1 = np.zeros( [ self.norbs_tot, self.norbs_tot ], dtype=int )
                            H1[ jumpsquare + row, jumpsquare + col ] = 1
                            H1[ jumpsquare + col, jumpsquare + row ] = 1
                            #theH1.append( H1 )
                            totalsize, H1start, H1row, H1col = sparsify (H1, totalsize, H1start, H1row, H1col)
                    jumpsquare += frag.norbs_frag
        H1start = np.array( H1start, dtype=ctypes.c_int )
        H1row   = np.array( H1row,   dtype=ctypes.c_int )
        H1col   = np.array( H1col,   dtype=ctypes.c_int )
        return ( H1start, H1row, H1col )
        
    def doexact( self, chempot_frag=0.0 ):
        oneRDM_loc = self.helper.construct1RDM_loc( self.doSCF, self.umat ) 
        self.energy = 0.0												
        self.spin = 0.0

        for frag in self.fragments:
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
        
            if ( self.do1EMB ):
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
        print ("Current sum of fragment energies: {0:.6f}".format (self.energy))
        print ("Current sum of fragment spins: {0:.6f}".format (self.spin))
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
            if self.doLASSCF:
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
            raise RuntimeError ("No translational invariance until you fix it!")
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
        self.energy = self.ints.e_tot
        iteration = 0
        u_diff = 1.0
        convergence_threshold = 1e-6
        rdm = np.zeros ((self.norbs_tot, self.norbs_tot))
        print ("RHF energy =", self.ints.fullEhf)

        while (u_diff > convergence_threshold):
            u_diff, rdm = self.doselfconsistent_corrpot (rdm, [('corrpot', iteration)])
            iteration += 1 
            if iteration > self.corrpot_maxiter:
                raise RuntimeError ('Maximum correlation-potential cycles!')

        if self.do1EMB:
            assert( len (self.fragments) == 1 )		
            print("-----NOTE: CASCI or Single embedding is used-----")				
            self.energy = self.fragments[0].E_imp
        if self.debug_energy:
            debug_Etot (self)

        for frag in self.fragments:
            if not (frag.imp_solver_name == 'dummy RHF'):
                fmt_str = "Writing {}".format (frag.frag_name) + " {} orbital molden"
                print (fmt_str.format ('natural'))
                frag.impurity_molden ('natorb', natorb=True)
                print (fmt_str.format ('impurity'))
                frag.impurity_molden ('imporb')
                print (fmt_str.format ('molecular'))
                frag.impurity_molden ('molorb', molorb=True)
        rdm = self.transform_ed_1rdm (get_od=True)
        no_occs, no_evecs = matrix_eigen_control_options (rdm, sort_vecs=-1, only_nonzero_vals=False)
        ao2no, no_ene, no_occ = self.get_las_nos (aobasis=True, oneRDM_loc=rdm, jmol_shift=True, try_symmetrize=True)
        print ("Whole-molecule natural orbital occupancies:\n{}".format (no_occ))
        print ("Writing trial wave function natural orbital molden")
        molden.from_mo (self.ints.mol, self.calcname + '_natorb.molden', ao2no, occ=no_occ, ene=no_ene)
        
        return self.energy

    def doselfconsistent_corrpot (self, rdm_old, iters):
        umat_old = np.array(self.umat, copy=True)
        
        # Find the chemical potential for the correlated impurity problem
        myiter = iters[-1][-1]
        nextiter = 0
        orb_diff = 1.0
        convergence_threshold = 1e-5
        while (np.any (np.asarray (orb_diff) > convergence_threshold)):
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
        if self.do1EMB:
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

        if not self.doLASSCF: self.save_checkpoint (self.calcname + '.chk.npy')

        return u_diff, rdm_new

    def doselfconsistent_orbs (self, iters):

        loc2wmas_old = np.concatenate ([frag.loc2amo for frag in self.fragments], axis=1)
        if self.doLASSCF and self.ints.symmetry and not is_subspace_block_adapted (loc2wmas_old, self.ints.loc2symm):
            print ("Active orbitals break symmetry :(")
            self.ints.symmetry = False
        try:
            loc2wmcs_old = get_complementary_states (loc2wmas_old, symm_blocks=self.ints.loc2symm)
        except linalg.LinAlgError as e:
            print (np.dot (loc2wmas_old.T, loc2wmas_old))
            raise (e)

        if self.doLASSCF:
            print ("Entering setup_wm_core_scf")
            self.ints.setup_wm_core_scf (self.fragments, self.calcname)
            self.align_symmetry ()
            self.save_checkpoint (self.calcname + '.chk.npy')

        oneRDM_loc = self.helper.construct1RDM_loc( self.doSCF, self.umat )
        if self.doLASSCF:
            print ("Entering refragmentation")
            oneRDM_loc = self.refragmentation (loc2wmas_old, loc2wmcs_old, oneRDM_loc)
        else:
            for frag in self.fragments:
                frag.restore_default_embedding_basis ()

        old_energy = self.energy
        self.energy = 0.0
        self.spin = 0.0
        for frag in self.fragments:
            print ("Entering Schmidt decomposition for {}".format (frag.frag_name))
            t0 = time.time ()
            frag.do_Schmidt (oneRDM_loc, self.fragments, loc2wmcs_old, self.doLASSCF)
            t1 = time.time ()
            print ("Entering impurity Hamiltonian construction for {}".format (frag.frag_name))
            frag.construct_impurity_hamiltonian ()
            t2 = time.time ()
            print ("Schmidt decomposition: {} seconds; impurity Hamiltonian construction: {} seconds".format (t1-t0, t2-t1))
        if self.examine_ifrag_olap:
            examine_ifrag_olap (self)
        if self.examine_wmcs:
            examine_wmcs (self)

        for itertype, iteridx in iters:
            print ("{0} iteration {1}".format (itertype, iteridx))

        # Iterate on chemical potential
        if self.do1EMB or self.do0SHOT or self.doLASSCF:
            self.chempot = 0.0
            self.doexact (self.chempot)
        else:
            self.chempot = optimize.newton( self.numeleccostfunction, self.chempot, tol=self.chempot_tol )
            print ("   Chemical potential =", self.chempot)
        #for frag in self.fragments:
            #frag.impurity_molden ('natorb', natorb=True)
            #frag.impurity_molden ('imporb')
            #frag.impurity_molden ('molorb', molorb=True)
        
        loc2wmas_new = np.concatenate ([frag.loc2amo for frag in self.fragments], axis=1)
        try:
            orb_diff = measure_basis_olap (loc2wmas_new, loc2wmcs_old)[0] / max (1,loc2wmas_new.shape[1])
        except:
            raise RuntimeError("what?\n{0}\n{1}".format(loc2wmas_new.shape,loc2wmcs_old.shape))

        oneRDM_locs = np.stack ([f.oneRDM_loc for f in self.fragments], axis=-1)
        energies = np.asarray ([f.E_imp for f in self.fragments])
        oneRDM_stdev = np.std (oneRDM_locs, axis=2)
        oneRDM_diff = np.linalg.norm (oneRDM_stdev)
        Eimp_stdev = np.std (energies)
        oneRDM_avg = self.transform_ed_1rdm (get_od=True)
        no_occs, no_evecs = matrix_eigen_control_options (oneRDM_avg, sort_vecs=-1, only_nonzero_vals=False)
        print ("Whole-molecule natural orbital occupancies:\n{}".format (no_occs))
        #ao2no = np.dot (self.ints.ao2loc, no_evecs)
        #molden.from_mo (self.ints.mol, self.calcname + '_natorb.molden', ao2no, occ=no_occs)
        print ("Whole-molecule 1RDM stdev norm = {}".format (oneRDM_diff))
        print ("Whole-molecule Eimp stdev = {}".format (Eimp_stdev))
        print ("Whole-molecule active-space orbital shift = {0}".format (orb_diff))
        if self.doLASSCF == False:
            orb_diff = oneRDM_diff = Eimp_stdev = Eiter = 0 # Do only 1 iteration
        else:
            self.energy = np.average (energies)
            Eiter = self.energy - old_energy
        print ("Whole-molecule energy difference = {}".format (Eiter))

        # Safety until I figure out how to deal with this degenerate-orbital thing
        if abs (Eiter) < 1e-7 and np.all (np.abs (energies - self.energy) < 1e-7):
            print ("Energies all converged to 100 nanoEh threshold; punking out of 1-RDM and orbital convergence")
            orb_diff = oneRDM_diff = 0
            

        return orb_diff, oneRDM_diff, Eimp_stdev, abs (Eiter)
        
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
            
    def transform_ed_1rdm( self, get_od=False ):
    
        norbs_frag = [f.loc2frag.shape[1] for f in self.fragments]
        result = np.zeros( [sum (norbs_frag), sum (norbs_frag)], dtype = np.float64)
        loc2frag = np.concatenate ([f.loc2frag for f in self.fragments], axis=1)
        assert (sum (norbs_frag) == loc2frag.shape[1])
        frag_ranges = [sum (norbs_frag[:i]) for i in range (len (norbs_frag) + 1)]
        for idx, f1 in enumerate (self.fragments):
            i = frag_ranges[idx]
            j = frag_ranges[idx+1]
            result[i:j,i:j] = f1.get_oneRDM_frag ()
            if get_od and idx+1 < len (self.fragments):
                for idx2, f2 in enumerate (self.fragments[idx+1:]):
                    k = frag_ranges[idx2+idx+1]
                    l = frag_ranges[idx2+idx+2]
                    result[i:j,k:l]  = represent_operator_in_basis (f1.oneRDM_loc, f1.loc2frag, f2.loc2frag)
                    result[i:j,k:l] += represent_operator_in_basis (f2.oneRDM_loc, f1.loc2frag, f2.loc2frag)
                    result[i:j,k:l] /= 2
                    result[k:l,i:j]  = result[i:j,k:l].T
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

        # Separately localize and assign the inactive and active orbitals as fragment orbitals
        loc2wmas = self.refrag_lowdin_active (loc2wmas, oneRDM_loc)
        loc2wmcs = self.refrag_lowdin_external (loc2wmcs, loc2wmas)

        # Evaluate how many electrons are in each active subspace
        interrs = []
        for loc2amo, frag in zip (loc2wmas, self.fragments):
            nelec_amo = np.einsum ("ip,ij,jp->",loc2amo.conjugate (), oneRDM_loc, loc2amo)
            interr = nelec_amo - round (nelec_amo)
            interrs.append (interr)
            print ("{0} fragment has {1:.5f} active electrons (integer error: {2:.2e})".format (frag.frag_name, nelec_amo, interr))
            #assert (interr < self.nelec_int_thresh), "Fragment with non-integer number of electrons appears"
            interr = np.eye (loc2amo.shape[1]) * interr / loc2amo.shape[1]
            oneRDM_loc -= reduce (np.dot, [loc2amo, interr, loc2amo.conjugate ().T])
        assert (all ((i < self.nelec_int_thresh for i in interrs))), "Fragment with non-integer number of electrons appears"
            
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
            if frag.imp_solver_name != 'dummy RHF':
                assert (is_basis_orthonormal (loc2imo))
                assert (is_basis_orthonormal (loc2amo))
                assert (is_basis_orthonormal (frag.loc2frag)), linalg.norm (loc2imo.conjugate ().T @ loc2amo)

        return oneRDM_loc

    def refrag_lowdin_active (self, loc2wmas, oneRDM_loc):
        
        ovlp_wmas = np.dot (loc2wmas.conjugate ().T, loc2wmas)
        oneRDM_amo = represent_operator_in_basis (oneRDM_loc, loc2wmas)
        print ("Trace of oneRDM_amo = {}".format (np.trace (oneRDM_amo)))

        #print ("Active orbital overlap matrix:\n{}".format (prettyprint (ovlp_wmas)))

        orth_idx = np.isclose (np.sum (ovlp_wmas - np.eye (ovlp_wmas.shape[0]), axis=1), 0)
        print ("{} active orbitals with overlap and {} active orbitals with no overlap".format (np.sum (~orth_idx), np.sum (orth_idx)))

        evecs = orth.lowdin (ovlp_wmas)        
        #print ("Orthonormal active orbitals eigenvectors:\n{}".format (prettyprint (evecs)))
        loc2wmas = np.dot (loc2wmas, evecs)

        for f in self.fragments:
            if f.loc2amo.shape[1] == 0:
                continue
            proj = represent_operator_in_basis (np.dot (f.loc2amo, f.loc2amo.conjugate ().T), loc2wmas)
            #print ("Projector matrix of {} active orbitals in orthonormal active basis:\n{}".format (
            #    f.frag_name, prettyprint (proj, fmt='{:5.2f}')))
        frag_weights = np.stack ([np.einsum ('ip,ij,jp->p', loc2wmas.conjugate (), np.dot (f.loc2amo, f.loc2amo.conjugate ().T), loc2wmas) for f in self.fragments], axis=-1)
        #print ("Fragment weights of orthonormal amos:\n{}".format (prettyprint (frag_weights, fmt='{:5.2f}')))
        idx = np.argsort (frag_weights, axis=1)[:,-1]

        loc2amo = [loc2wmas[:,(idx == i)] for i in range (len (self.fragments))]     
        nelec_amo = [np.einsum ('ip,ij,jp->', l.conjugate (), oneRDM_loc, l) for l in loc2amo]
        print ("Number of electrons in each fragment:")
        for f, n in zip (self.fragments, nelec_amo):
            print ("{}: {}".format (f.frag_name, n))
        
        print ("Entanglement norms:")
        for (idx1, f1), (idx2, f2) in combinations (enumerate (self.fragments), 2):
            print ("{}-{}: {}".format (f1.frag_name, f2.frag_name, np.linalg.norm (np.einsum ('ip,ij,iq->pq', loc2amo[idx1], oneRDM_loc, loc2amo[idx2]))))

        dm = represent_operator_in_basis (oneRDM_loc, np.concatenate (loc2amo, axis=1))
        #print ("Density matrix in symmetrically-orthogonalized active-orbital basis:\n{}".format (prettyprint (dm))) 
        for l in loc2amo:
            assert (is_basis_orthonormal (l))

        return loc2amo

    def refrag_lowdin_external (self, loc2wmcs, loc2wmas):

        proj_gfrag = np.stack ([np.dot (f.get_true_loc2frag (), f.get_true_loc2frag ().conjugate ().T) for f in self.fragments], axis=-1)
        loc2x = loc2wmcs.copy ()
        loc2wmcs = [np.zeros ((self.norbs_tot, 0), dtype=loc2x.dtype) for ix in range (len (self.fragments))]

        # First pass: Only retain the highest Mfk - Mak eigenvalues, for which the eigenvalue is larger than 1/2.
        # Do dummy fragments last so that distributing orbitals to them doesn't fuck up symmetry
        Mxk_tot = np.asarray ([f.norbs_frag - loc2wmas[ix].shape[1] for ix, f in enumerate (self.fragments)])
        for ix, f in enumerate (self.fragments):
            if f.imp_solver_name == 'dummy RHF': Mxk_tot[ix] = 0
        it = 0
        assign_thresh = 1/2
        Mxk_rem = Mxk_tot
        max_eval = np.ones (len (self.fragments))
        # Do NOT attempt to enforce symmetry here YET. It may be necessary in future, but for now hold off!
        while np.any (Mxk_rem): #loc2x.shape[1] > 0:
            if np.all (max_eval < assign_thresh):
                print (("Warning: external orbital assignment threshold lowered from {} to {}"
                    " after failing to assign any orbitals in iteration {}").format (assign_thresh, np.max (max_eval)*0.99, it-1))
                assign_thresh = np.max (max_eval)*0.99
            Mxk = np.asarray ([w.shape[1] for w in loc2wmcs])
            Mxk_rem = Mxk_tot - Mxk
            ix_rem = Mxk_rem > 0
            max_eval = np.zeros (len (self.fragments))
            proj_gfrag_sum = proj_gfrag[:,:,ix_rem].sum (2)
            for ix_frag, f in enumerate (self.fragments):
                if loc2x.shape[1] == 0 or Mxk_rem[ix_frag] == 0:
                    continue
                print ("it {}\nMxk = {}\nMxk_rem = {}\nix_rem = {}\nloc2x.shape = {}".format (it, Mxk, Mxk_rem, ix_rem, loc2x.shape))
                p = represent_operator_in_basis (proj_gfrag[:,:,ix_frag], loc2x)
                evals, evecs = matrix_eigen_control_options (p, sort_vecs=-1, only_nonzero_vals=False)
                ix_zero = np.abs (evals) < 1e-8
                loc2evecs = np.dot (loc2x, evecs)
                # Scale the eigenvalues to evaluate their comparison to the sum of projector expt vals of unfull fragments
                # exptvals = np.einsum ('ip,ijk,k,jp->p', loc2evecs.conjugate (), proj_gfrag, ix_rem.astype (int), loc2evecs)
                exptvals = ((proj_gfrag_sum @ loc2evecs) * loc2evecs).sum (0)
                evals = evals / exptvals
                evals[ix_zero] = 0
                max_eval[ix_frag] = np.max (evals)
                ix_loc = evals >= assign_thresh
                ix_loc[Mxk_rem[ix_frag]:] = False
                print ("{} fragment, iteration {}: {} eigenvalues above 1/2 found of {} total sought:\n{}".format (
                    f.frag_name, it, np.count_nonzero (ix_loc), Mxk_rem[ix_frag], evals))
                loc2wmcs[ix_frag] = np.append (loc2wmcs[ix_frag], loc2evecs[:,ix_loc], axis=1)
                loc2x = loc2evecs[:,~ix_loc]
            it += 1

        # Just assign (potentially symmetry-breaking!) orbitals to dummy fragments randomly because it cannot possibly matter
        for ix, f in enumerate (self.fragments):
            if f.imp_solver_name != 'dummy RHF':
                continue
            loc2wmcs[ix] = loc2x[:,:f.norbs_frag]
            loc2x = loc2x[:,f.norbs_frag:]


        for ix, loc2frag in enumerate (loc2wmcs):
            ovlp = loc2frag.conjugate ().T
            ovlp = np.dot (ovlp, loc2frag)
            err = linalg.norm (ovlp - np.eye (ovlp.shape[0]))
            it = 0
            while (abs (err) > 1e-8):
                umat = orth.lowdin (ovlp)
                loc2frag = np.dot (loc2frag, umat)
                ovlp = loc2frag.conjugate ().T
                ovlp = np.dot (ovlp, loc2frag)
                err = linalg.norm (ovlp - np.eye (ovlp.shape[0]))
                it += 1
                if it > 100:
                    raise RuntimeError ("I tried 100 times to orthonormalize this and failed")
            loc2wmcs[ix] = loc2frag

        #for loc2, f in zip (loc2wmcs, self.fragments):
        #    print ("Projector eigenvalues of {} external fragment orbitals:".format (f.frag_name))
        #    frag_weights = np.einsum ('ip,ijq,jp->pq', loc2.conjugate (), proj_gfrag, loc2)
        #    print (prettyprint (frag_weights))
             
        for l in loc2wmcs:
            assert (is_basis_orthonormal (l))
        return loc2wmcs

    def generate_frag_cas_guess (self, mo_coeff=None, caslst=None, cas_irrep_nocc=None, cas_irrep_ncore=None, replace=False, force_imp=False, confine_guess=True, guess_somos = []):
        ''' Generate initial active-orbital guesses for fragments from the whole-molecule MOs. Either add active orbitals
        or replace them. '''
        
        if replace:
            print ("Deleting existing amos...")
            for f in self.fragments:
                f.loc2amo = np.zeros ((self.ints.norbs_tot, 0), dtype=self.ints.ao2loc.dtype)
                f.oneRDMas_loc = np.zeros_like (self.ints.oneRDM_loc)
                f.twoCDMimp_amo = np.zeros ((0,0,0,0), dtype=self.ints.ao2loc.dtype)
                f.ci_as = None
        loc2wmas_current = np.concatenate ([f.loc2amo for f in self.fragments if f.active_space is not None], axis=1)
        if mo_coeff is None:        
            if self.doLASSCF:
                mo_coeff = self.ints.get_trial_nos (ao_basis=True, loc2wmas=loc2wmas_current)[0]
            else:
                mo_coeff = self.ints.get_trial_nos (ao_basis=True, loc2wmas=None)[0]
        
        nelec_wmas_target = sum ([f.active_space[0] for f in self.fragments if f.active_space is not None])
        norbs_wmas_target = sum ([f.active_space[1] for f in self.fragments if f.active_space is not None])
        assert ((self.ints.nelec_tot - nelec_wmas_target) % 2 == 0), 'parity problem: {} of {} electrons active in target'.format (nelec_wmas_target, self.ints.nelec_tot)
        ncore_target = (self.ints.nelec_tot - nelec_wmas_target) // 2
        nocc_target = ncore_target + norbs_wmas_target
        print ("ncore_target = {}; nocc_target = {}".format (ncore_target, nocc_target))

        nelec_wmas_current = sum ([f.nelec_as for f in self.fragments if f.norbs_as > 0])
        norbs_wmas_current = sum ([f.norbs_as for f in self.fragments if f.norbs_as > 0])
        assert ((self.ints.nelec_tot - nelec_wmas_current) % 2 == 0), 'parity problem: {} of {} electrons active in currently'.format (nelec_wmas_target, self.ints.nelec_tot)
        ncore_current = (self.ints.nelec_tot - nelec_wmas_current) // 2
        nocc_current = ncore_current + norbs_wmas_current
        print ("ncore_current = {}; nocc_current = {}".format (ncore_current, nocc_current))

        if norbs_wmas_current > 0:
            wmas2ao_current = (self.ints.ao2loc @ loc2wmas_current).conjugate ().T
            amo_coeff = mo_coeff[:,ncore_current:nocc_current]
            ovlp = wmas2ao_current @ self.ints.ao_ovlp @ amo_coeff
            ovlp = np.trace (ovlp.conjugate ().T @ ovlp) / norbs_wmas_current
            assert (np.isclose (ovlp, 1)), 'unless replace=True, mo_coeff must contain current active orbitals at range {}:{} (err={})'.format (ncore_current,nocc_current, ovlp-1)
            if caslst is not None and len (caslst) > 0:
                assert (np.all (np.isin (list(range(ncore_current+1,nocc_current+1)), caslst))), 'caslst must contain range {}:{} inclusive ({})'.format (ncore_current+1, nocc_current, caslst)

        if caslst is not None and len (caslst) == 0: caslst = None
        if caslst is not None and cas_irrep_nocc is not None:
            print ("Warning: caslst overrides cas_irrep_nocc!")
        if caslst is not None:
            mf = scf.RHF (self.ints.mol)
            mf.mo_coeff = mo_coeff
            cas = mcscf.CASCI (mf, norbs_wmas_target, nelec_wmas_target)
            mo_coeff = cas.sort_mo (caslst)
        elif cas_irrep_nocc is not None:
            mf = scf.RHF (self.ints.mol)
            mf.mo_coeff = mo_coeff
            cas = mcscf.CASCI (mf, norbs_wmas_target, nelec_wmas_target)
            mo_coeff = cas.sort_mo_by_irrep (cas_irrep_nocc, cas_irrep_ncore=cas_irrep_ncore)
            
            
        amo_new_coeff = np.append (mo_coeff[:,ncore_target:ncore_current], mo_coeff[:,nocc_current:nocc_target], axis=1)

        loc2amo_new = orthonormalize_a_basis (linalg.solve (self.ints.ao2loc, amo_new_coeff))
        if self.ints.symmetry:
            if is_subspace_block_adapted (loc2amo_new, self.ints.loc2symm):
                loc2amo_new, labels = symmetrize_basis (loc2amo_new, self.ints.loc2symm)[:2]
                print ("Guess active orbital irreps: {}".format (labels))
            else:
                print ("NOTE: symmetric system but guess active orbital space is NOT symmetry-adapted!")
        projamo = np.dot (loc2amo_new, loc2amo_new.conjugate ().T)

        for f in self.fragments:
            if f.active_space is not None:
                my_new_orbs = f.active_space[1] - f.norbs_as
                if confine_guess:
                    p = represent_operator_in_basis (proj_amo, f.get_true_loc2frag ())
                    evals, evecs = matrix_eigen_control_options (p, sort_vecs=-1, only_nonzero_vals=False)
                    evals = evals[:f.active_space[1]]
                    loc2amo_guess = np.dot (f.get_true_loc2frag (), evecs[:,:my_new_orbs])
                else:
                    proj_frag = np.dot (f.get_true_loc2frag (), f.get_true_loc2frag ().conjugate ().T)
                    proj_frag = represent_operator_in_basis (proj_frag, loc2amo_new)
                    evals, evecs = matrix_eigen_control_options (proj_frag, sort_vecs=-1, only_nonzero_vals=False)
                    loc2amo_guess = np.dot (loc2amo_new, evecs[:,:my_new_orbs]) 
                # Set up somo twoCDM for accurate initial guess energy
                evals, evecs = matrix_eigen_control_options (represent_operator_in_basis (self.ints.oneRDM_loc, loc2amo_guess))
                idx_somo = (evals > 0.8) & (evals < 1.2)
                nsomo = np.count_nonzero (idx_somo)
                if nsomo > 0:
                    loc2somo = loc2amo_guess @ evecs[:,idx_somo]
                    dma = np.eye (nsomo)
                    dmb = np.zeros ((nsomo, nsomo), dtype=dma.dtype)
                    twoCDM_somo = get_2CDM_from_2RDM (get_2RDM_from_2CDM (np.zeros ([nsomo,]*4, dtype=dma.dtype), [dma,dmb]), dma+dmb)
                # Back to the guess orbitals
                loc2amo_guess = np.append (f.loc2amo, loc2amo_guess, axis=1)
                fock = represent_operator_in_basis (self.ints.activeFOCK, loc2amo_guess)
                ovlp = loc2amo_guess.conjugate ().T @ loc2amo_guess
                evals, evecs = linalg.eigh (fock, b=ovlp)
                idx = np.argsort (evals)
                f.loc2amo_guess = np.dot (loc2amo_guess, evecs[:,idx])
                if force_imp:
                    print ("force_imp!")
                    old2new_amo = f.loc2amo.conjugate ().T @ f.loc2amo_guess
                    f.loc2amo = f.loc2amo_guess.copy ()
                    f.oneRDMas_loc = project_operator_into_subspace (self.ints.oneRDM_loc, f.loc2amo)
                    f.twoCDMimp_amo = represent_operator_in_basis (f.twoCDMimp_amo, old2new_amo)
                    if nsomo > 0:
                        somo2amo = loc2somo.conjugate ().T @ f.loc2amo
                        f.twoCDMimp_amo += represent_operator_in_basis (twoCDM_somo, somo2amo)
                    f.ci_as = None

        if len (guess_somos) == len (self.fragments):
            assert (not force_imp), "Don't force_imp and guess_somos at the same time"
            # construct rohf-like density matrices and set loc2amo_guess -> loc2amo
            for nsomo, f, in zip (guess_somos, self.fragments):
                if f.active_space is None:
                    continue
                assert (nsomo % 2 == f.active_space[0] % 2)
                assert (nsomo <= f.active_space[1])
                neleca = (f.active_space[0] + nsomo) // 2
                nelecb = (f.active_space[0] - nsomo) // 2
                occa = np.zeros (f.active_space[1], dtype=np.float64)
                occb = np.zeros (f.active_space[1], dtype=np.float64)
                occa[:neleca] += 1
                occb[:nelecb] += 1
                dma, dmb = np.diag (occa), np.diag (occb)
                # This is to cajole my ROHF-like pseudo-cumulant decomposition
                twoCDM = np.zeros ([f.active_space[1] for i in range (4)], dtype=np.float64)
                twoRDM = get_2RDM_from_2CDM (twoCDM, [dma, dmb])
                dm = dma + dmb
                f.loc2amo = f.loc2amo_guess.copy ()
                f.twoCDMimp_amo = get_2CDM_from_2RDM (twoRDM, dm) 
                f.oneRDMas_loc = represent_operator_in_basis (dm, f.loc2amo.conjugate ().T)
                f.ci_as = None
            
    def save_checkpoint (self, fname):
        ''' Data array structure: nao_nr, chempot, 1RDM or umat, norbs_amo in frag 1, loc2amo of frag 1, oneRDM_amo of frag 1, twoCDMimp_amo of frag 1, norbs_amo of frag 2, ... '''
        nao = self.ints.mol.nao_nr ()
        if self.doLASSCF:
            chkdata = self.helper.construct1RDM_loc (self.doSCF, self.umat)
            chkdata = represent_operator_in_basis (chkdata, self.ints.ao2loc.conjugate ().T).flatten (order='C')
        else:
            chkdata = represent_operator_in_basis (self.umat, self.ints.ao2loc.conjugate ().T).flatten (order='C')
        chkdata = np.append (np.asarray ([nao, self.chempot]), chkdata)
        for f in self.fragments:
            chkdata = np.append (chkdata, [f.norbs_as])
            chkdata = np.append (chkdata, np.dot (self.ints.ao2loc, f.loc2amo).flatten (order='C'))
            chkdata = np.append (chkdata, represent_operator_in_basis (f.oneRDM_loc, f.loc2amo).flatten (order='C'))
            chkdata = np.append (chkdata, f.twoCDMimp_amo.flatten (order='C'))
        np.save (fname, chkdata)

    def load_checkpoint (self, fname, prev_mol=None):
        nelec_amo = sum ((f.active_space[0] for f in self.fragments if f.active_space is not None))
        norbs_amo = sum ((f.active_space[1] for f in self.fragments if f.active_space is not None))
        norbs_cmo = (self.ints.mol.nelectron - nelec_amo) // 2
        norbs_omo = norbs_cmo + norbs_amo
        chkdata = np.load (fname)
        nao, self.chempot, chkdata = int (round (chkdata[0])), chkdata[1], chkdata[2:] 
        print ("{} atomic orbital basis functions reported in checkpoint file, as opposed to {} in integral object".format (nao, self.ints.mol.nao_nr ()))
        assert (prev_mol is not None or nao == self.ints.mol.nao_nr ())
        locSao = np.dot (self.ints.ao_ovlp, self.ints.ao2loc).conjugate ().T
        if prev_mol:
            oldSnew = mole.intor_cross('int1e_ovlp', prev_mol, self.ints.mol)
            aoSloc = np.dot (oldSnew, self.ints.ao2loc)
            if same_mol (prev_mol, self.ints.mol, cmp_basis = False): #basis set expansion
                locSao = np.dot (self.ints.ao_ovlp, self.ints.ao2loc).conjugate ().T
            else: # geometry change
                locSao = aoSloc.conjugate ().T
        else:
            aoSloc = np.dot (self.ints.ao_ovlp, self.ints.ao2loc)
            locSao = aoSloc.conjugate ().T

        mat, chkdata = chkdata[:nao**2].reshape (nao, nao, order='C'), chkdata[nao**2:]
        mat = represent_operator_in_basis (mat, aoSloc)
        if self.doLASSCF:
            self.ints.oneRDM_loc = mat.copy ()
            if prev_mol is not None:
                assert (abs (np.trace (self.ints.oneRDM_loc) - self.ints.nelec_tot) < 1e-8), "checkpoint oneRDM trace = {}; nelec_tot = {}".format (
                    np.trace (self.ints.oneRDM_loc), self.ints.nelec_tot)
        else:
            self.umat = mat.copy ()

        for f in self.fragments:
            if self.doLASSCF: f.oneRDM_loc = self.ints.oneRDM_loc
            namo, chkdata = int (round (chkdata[0])), chkdata[1:]
            print ("{} active orbitals reported in checkpoint file for fragment {}".format (namo, f.frag_name))
            if namo > 0:
                f.loc2amo,       chkdata = chkdata[:nao*namo].reshape (nao, namo, order='C'), chkdata[nao*namo:]
                f.oneRDMas_loc,  chkdata = chkdata[:namo**2].reshape (namo, namo, order='C'), chkdata[namo**2:]
                print ("{} fragment oneRDM_amo (trace = {}):\n{}".format (
                    f.frag_name, np.trace (f.oneRDMas_loc), prettyprint (f.oneRDMas_loc, fmt='{:6.3f}')))
                f.twoCDMimp_amo, chkdata = chkdata[:namo**4].reshape (namo, namo, namo, namo, order='C'), chkdata[namo**4:]
                if prev_mol and same_mol (prev_mol, self.ints.mol, cmp_basis=False): f.loc2amo = project_mo_nr2nr (prev_mol, f.loc2amo, self.ints.mol)
                f.loc2amo = np.dot (locSao, f.loc2amo)
                ovlp = np.dot (f.loc2amo.conjugate ().T, f.loc2amo)
                evecs = orth.lowdin (ovlp)
                f.loc2amo = np.dot (f.loc2amo, evecs)
                f.twoCDMimp_amo = represent_operator_in_basis (f.twoCDMimp_amo, evecs)
                # natorbify
                no_occ, no_evecs = matrix_eigen_control_options (f.oneRDMas_loc, sort_vecs=-1, only_nonzero_vals=False)
                f.loc2amo = np.dot (f.loc2amo, no_evecs)
                f.oneRDMas_loc = represent_operator_in_basis (np.diag (no_occ), f.loc2amo.conjugate ().T)
                f.twoCDMimp_amo = represent_operator_in_basis (f.twoCDMimp_amo, no_evecs)
                print ("{} fragment oneRDM_amo (trace = {}):\n{}".format (
                    f.frag_name, np.trace (f.oneRDMas_loc), prettyprint (represent_operator_in_basis (f.oneRDMas_loc, f.loc2amo), fmt='{:6.3f}')))
               
                if np.amax (np.abs (f.twoCDMimp_amo)) > 1e-10:
                    tei = self.ints.dmet_tei (f.loc2amo)
                    f.E2_cum = np.tensordot (tei, f.twoCDMimp_amo, axes=4) / 2
        assert (chkdata.shape == tuple((0,))), chkdata.shape               

        if self.doLASSCF: self.ints.setup_wm_core_scf (self.fragments, self.calcname)
        
    def expand_active_space (self, callables):
        ''' Add occupied or vitual orbitals from the whole molecule to one or several fragments' active spaces using functions in list "callables"
            Callables should have signature (self.ints.mol, mo, myamo, ncore, ncas) where mo is mo coefficients and myamo is the active orbital
            coefficients of the current fragment. They should return the expanded set of active orbital coefficients.
            All orbital coefficients interacting with the callables are in the ao basis (not the orthonormal localized DMET/LASSCF basis)

            Keeping the additional orbitals localized is your responsibility '''

        loc2wmas = np.concatenate ([frag.loc2amo for frag in self.fragments], axis=1)
        mo = self.ints.get_trial_nos (aobasis=True, loc2wmas=loc2wmas)[0]

        for frag, cal in zip (self.fragments, callables):
            if frag.active_space is None:
                continue
            myamo = cal (self.ints.mol, mo, np.dot (self.ints.ao2loc, frag.loc2amo), norbs_cmo, norbs_amo)
            old2new_amo = frag.loc2amo.conjugate ().T.copy ()
            frag.loc2amo = reduce (np.dot, (self.ints.ao2loc.conjugate ().T, self.ints.ao_ovlp, myamo))
            old2new_amo = np.dot (old2new_amo, frag.loc2amo)
            frag.oneRDMas_loc = project_operator_into_subspace (self.ints.oneRDM_loc, frag.loc2amo) 
            frag.twoCDMimp_amo = represent_operator_in_basis (frag.twoCDMimp_amo, old2new_amo) 


    def get_las_nos (self, **kwargs):
        ''' Save MOs in a form that can be loaded for a CAS calculation on a npy file '''
        kwargs['aobasis'] = True
        kwargs['loc2wmas'] = np.concatenate ([frag.loc2amo for frag in self.fragments], axis=1)
        return self.ints.get_trial_nos (**kwargs)

    def align_symmetry (self):
        ''' Check if the LAS wave function is (still) point-group-symmetry adapted and symmetrize the active orbitals if so '''
        symmetry, ir_names = self.examine_symmetry (verbose=False)
        if symmetry:
            fake_mol = type ('', (), {'symm_orb': self.ints.loc2symm}) ()
            for f in self.fragments:
                f.symmetry = symmetry
                f.ir_names = ir_names
                if not f.norbs_as:
                    continue
                old2loc = f.loc2amo.conjugate ().T.copy ()
                f.loc2amo = symmetrize_space (fake_mol, f.loc2amo, s=np.eye (self.norbs_tot))
                old2new = old2loc @ f.loc2amo
                print (old2new)
                f.twoCDMimp_amo = represent_operator_in_basis (f.twoCDMimp_amo, old2new)

    def examine_symmetry (self, verbose=True):
        if not self.ints.mol.symmetry:
            return False, None, None
        symmetry = self.ints.mol.groupname
        def symm_analysis (mo, mo_name, symmetry):
            if verbose: print ("Analyzing {} fragment orbitals:".format (mo_name))
            irrep_norbs = []
            for irrep, loc2ir in zip (self.ints.mol.irrep_name, self.ints.loc2symm):
                ovlp = mo.conjugate ().T @ loc2ir
                irrep_norbs.append (np.trace (ovlp @ ovlp.conjugate ().T))
                if verbose: print ("{}: {}".format (irrep, irrep_norbs[-1]))
            if verbose: print ("Total (sanity check!): {}".format (sum (irrep_norbs)))
            adapted = np.allclose (irrep_norbs, np.around (irrep_norbs))
            if not adapted: symmetry = False
            return symmetry, irrep_norbs

        loc2dummy = np.concatenate ([f.loc2frag for f in self.fragments if f.imp_solver_name == 'dummy RHF'], axis=1)
        symmetry = symm_analysis (loc2dummy, 'dummy', symmetry)[0]
        irrep_norbs = np.zeros ((len (self.fragments), len (self.ints.mol.irrep_name)), dtype=np.int32)
        for idx, f in enumerate (self.fragments):
            if f.imp_solver_name == 'dummy RHF':
                continue
            symmetry, irrep_norbs[idx,:] = symm_analysis (f.loc2frag, f.frag_name, symmetry)

        nelec_symm = [np.trace (represent_operator_in_basis (self.ints.oneRDM_loc, loc2ir)) for loc2ir in self.ints.loc2symm]
        if verbose:
            print ("Number of electrons in each irrep of trial wvfn:")
            for ir, n in zip (self.ints.mol.irrep_name, nelec_symm):
                print ("{}: {}".format (ir, n))
        # I think these are actually allowed to be non-integers. It's the coupling elements that have to be zero because an element
        # of the density operator that couples orbitals of two irreps necessarily couples CI vector elements of two irreps
        # (something something Abelian groups)
        for (ir1, loc2ir1), (ir2, loc2ir2) in combinations (zip (self.ints.mol.irrep_name, self.ints.loc2symm), 2):
            off = np.amax (np.abs (represent_operator_in_basis (self.ints.oneRDM_loc, loc2ir1, loc2ir2)))
            if verbose: print ("1RDM maximum element coupling {} to {}: {}".format (ir1, ir2, off))
            if off > 1e-7: symmetry = False

        # In principle I should also check the twoCDM, but it ends up not mattering because the twoCDM only contributes
        # a constant to the energy!

        if symmetry:
            print ("This system retains {} symmetry!".format (symmetry))
        else:
            print ("This system loses its symmetry")
        return symmetry, self.ints.mol.irrep_name



