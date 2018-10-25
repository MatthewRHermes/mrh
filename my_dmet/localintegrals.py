'''
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

#import qcdmet_paths
from pyscf import gto, scf, ao2mo, tools, lo
from pyscf.lo import nao, orth, boys
from pyscf.tools import molden
from pyscf.lib import numpy_helper as pyscf_np
from mrh.my_dmet import rhf as wm_rhf
from mrh.my_dmet import iao_helper
import numpy as np
import scipy
from mrh.util.my_math import is_close_to_integer
from mrh.util.rdm import get_1RDM_from_OEI
from mrh.util.basis import represent_operator_in_basis, get_complementary_states, project_operator_into_subspace, orthonormalize_a_basis, get_overlapping_states
from mrh.util.tensors import symmetrize_tensor
from mrh.util.la import matrix_eigen_control_options, is_matrix_eye
from mrh.util import params
from math import sqrt
import itertools
from functools import reduce

class localintegrals:

    def __init__( self, the_mf, active_orbs, localizationtype, ao_rotation=None, use_full_hessian=True, localization_threshold=1e-6 ):

        assert (( localizationtype == 'meta_lowdin' ) or ( localizationtype == 'boys' ) or ( localizationtype == 'lowdin' ) or ( localizationtype == 'iao' ))
        self.num_mf_stab_checks = 0
        
        # Information on the full HF problem
        self.mol        = the_mf.mol
        self.fullovlpao = the_mf.get_ovlp
        self.fullEhf    = the_mf.e_tot
        self.fullDMao   = np.dot(np.dot( the_mf.mo_coeff, np.diag( the_mf.mo_occ )), the_mf.mo_coeff.T )
        self.fullJKao   = scf.hf.get_veff( self.mol, self.fullDMao, 0, 0, 1 ) #Last 3 numbers: dm_last, vhf_last, hermi
        self.fullFOCKao = self.mol.intor('cint1e_kin_sph') + self.mol.intor('cint1e_nuc_sph') + self.fullJKao
        
        # Active space information
        self._which    = localizationtype
        self.active    = np.zeros( [ self.mol.nao_nr() ], dtype=int )
        self.active[ active_orbs ] = 1
        self.norbs_tot = np.sum( self.active ) # Number of active space orbitals
        self.nelec_tot = int(np.rint( self.mol.nelectron - np.sum( the_mf.mo_occ[ self.active==0 ] ))) # Total number of electrons minus frozen part
        
        # Localize the orbitals
        if (( self._which == 'meta_lowdin' ) or ( self._which == 'boys' )):
            if ( self._which == 'meta_lowdin' ):
                assert( self.norbs_tot == self.mol.nao_nr() ) # Full active space required
            if ( self._which == 'boys' ):
                self.ao2loc = the_mf.mo_coeff[ : , self.active==1 ]
            if ( self.norbs_tot == self.mol.nao_nr() ): # If you want the full active, do meta-Lowdin
                nao.AOSHELL[4] = ['1s0p0d0f', '2s1p0d0f'] # redefine the valence shell for Be
                self.ao2loc = orth.orth_ao( self.mol, 'meta_lowdin' )
                if ( ao_rotation != None ):
                    self.ao2loc = np.dot( self.ao2loc, ao_rotation.T )
            if ( self._which == 'boys' ):
                old_verbose = self.mol.verbose
                self.mol.verbose = 5
                loc = boys.Boys (self.mol, self.ao2loc)
#                loc = localizer.localizer( self.mol, self.ao2loc, self._which, use_full_hessian )
                self.mol.verbose = old_verbose
#                self.ao2loc = loc.optimize( threshold=localization_threshold )
                self.ao2loc = loc.kernel ()
            self.TI_OK = False # Check yourself if OK, then overwrite
        if ( self._which == 'lowdin' ):
            assert( self.norbs_tot == self.mol.nao_nr() ) # Full active space required
            ovlp = self.mol.intor('cint1e_ovlp_sph')
            ovlp_eigs, ovlp_vecs = np.linalg.eigh( ovlp )
            assert ( np.linalg.norm( np.dot( np.dot( ovlp_vecs, np.diag( ovlp_eigs ) ), ovlp_vecs.T ) - ovlp ) < 1e-10 )
            self.ao2loc = np.dot( np.dot( ovlp_vecs, np.diag( np.power( ovlp_eigs, -0.5 ) ) ), ovlp_vecs.T )
            self.TI_OK  = False # Check yourself if OK, then overwrite
        if ( self._which == 'iao' ):
            assert( self.norbs_tot == self.mol.nao_nr() ) # Full active space assumed
            self.ao2loc = iao_helper.localize_iao( self.mol, the_mf )
            if ( ao_rotation != None ):
                self.ao2loc = np.dot( self.ao2loc, ao_rotation.T )
            self.TI_OK = False # Check yourself if OK, then overwrite
            #self.molden( 'dump.molden' ) # Debugging mode
        assert( self.loc_ortho() < 1e-8 )

        # Stored inverse overlap matrix
        self.ao_ovlp_inv = np.dot (self.ao2loc, self.ao2loc.conjugate ().T)
        self.ao_ovlp     = the_mf.get_ovlp ()
        assert (is_matrix_eye (np.dot (self.ao_ovlp, self.ao_ovlp_inv)))
        
        # Effective Hamiltonian due to frozen part
        self.frozenDMmo  = np.array( the_mf.mo_occ, copy=True )
        self.frozenDMmo[ self.active==1 ] = 0 # Only the frozen MO occupancies nonzero
        self.frozenDMao  = np.dot(np.dot( the_mf.mo_coeff, np.diag( self.frozenDMmo )), the_mf.mo_coeff.T )
        self.frozenJKao  = scf.hf.get_veff( self.mol, self.frozenDMao, 0, 0, 1 ) #Last 3 numbers: dm_last, vhf_last, hermi
        self.frozenOEIao = self.fullFOCKao - self.fullJKao + self.frozenJKao
        
        # Localized OEI and ERI
        self.activeCONST    = self.mol.energy_nuc() + np.einsum( 'ij,ij->', self.frozenOEIao - 0.5*self.frozenJKao, self.frozenDMao )
        self.activeOEI      = represent_operator_in_basis (self.frozenOEIao, self.ao2loc )
        self.activeFOCK     = represent_operator_in_basis (self.fullFOCKao,  self.ao2loc )
        self.activeJKidem   = self.activeFOCK - self.activeOEI
        self.activeJKcorr   = np.zeros ((self.norbs_tot, self.norbs_tot), dtype=self.activeOEI.dtype)
        self.oneRDMcorr_loc = np.zeros ((self.norbs_tot, self.norbs_tot), dtype=self.activeOEI.dtype)
        self.loc2idem       = np.eye (self.norbs_tot, dtype=self.activeOEI.dtype)
        self.nelec_idem     = self.nelec_tot
        self.ERIinMEM       = False
        self.activeERI      = None
        self.idemERI        = None
        if ( self.norbs_tot <= 125 ):
            self.ERIinMEM   = True
            self.activeERI  = ao2mo.outcore.full_iofree( self.mol, self.ao2loc, compact=False ).reshape(self.norbs_tot, self.norbs_tot, self.norbs_tot, self.norbs_tot)
            self.idemERI    = self.activeERI

    def molden( self, filename ):
    
        with open( filename, 'w' ) as thefile:
            molden.header( self.mol, thefile )
            molden.orbital_coeff( self.mol, thefile, self.ao2loc )
            
    def loc_ortho( self ):
    
#        ShouldBeI = np.dot( np.dot( self.ao2loc.T , self.mol.intor('cint1e_ovlp_sph') ) , self.ao2loc )
        ShouldBeI = represent_operator_in_basis (self.fullovlpao (), self.ao2loc )
        return np.linalg.norm( ShouldBeI - np.eye( ShouldBeI.shape[0] ) )
        
    def debug_matrixelements( self ):
    
        eigvals, eigvecs = np.linalg.eigh( self.activeFOCK )
        eigvecs = eigvecs[ :, eigvals.argsort() ]
        assert( self.nelec_tot % 2 == 0 )
        numPairs = self.nelec_tot // 2
        DMguess = 2 * np.dot( eigvecs[ :, :numPairs ], eigvecs[ :, :numPairs ].T )
        if ( self.ERIinMEM == True ):
            DMloc = wm_rhf.solve_ERI( self.activeOEI, self.activeERI, DMguess, numPairs, num_mf_stab_checks )
        else:
            DMloc = wm_rhf.solve_JK( self.activeOEI, self.mol, self.ao2loc, DMguess, numPairs )
        newFOCKloc = self.loc_rhf_fock_bis( DMloc )
        newRHFener = self.activeCONST + 0.5 * np.einsum( 'ij,ij->', DMloc, self.activeOEI + newFOCKloc )
        print("2-norm difference of RDM(self.activeFOCK) and RDM(self.active{OEI,ERI})  =", np.linalg.norm( DMguess - DMloc ))
        print("2-norm difference of self.activeFOCK and FOCK(RDM(self.active{OEI,ERI})) =", np.linalg.norm( self.activeFOCK - newFOCKloc ))
        print("RHF energy of mean-field input           =", self.fullEhf)
        print("RHF energy based on self.active{OEI,ERI} =", newRHFener)
        
    def const( self ):
    
        return self.activeCONST

    def loc_oei( self ):

        return self.activeOEI + self.activeJKcorr
        
    def loc_rhf_fock( self ):

        return self.activeOEI + self.activeJKcorr + self.activeJKidem
        
    def loc_rhf_jk_bis( self, DMloc ):
    
        if ( self.ERIinMEM == False ):
            DM_ao = represent_operator_in_basis (DMloc, self.ao2loc.T )
            JK_ao = scf.hf.get_veff( self.mol, DM_ao, 0, 0, 1 ) #Last 3 numbers: dm_last, vhf_last, hermi
            JK_loc = represent_operator_in_basis (JK_ao, self.ao2loc )
        else:
            JK_loc = np.einsum( 'ijkl,ij->kl', self.activeERI, DMloc ) - 0.5 * np.einsum( 'ijkl,ik->jl', self.activeERI, DMloc )
        return JK_loc

    def loc_rhf_fock_bis( self, DMloc ):
   
        # I can't alter activeOEI because I don't want the meaning of this function to change 
        return self.activeOEI + self.loc_rhf_jk_bis (DMloc)

    def loc_tei( self ):
    
        if ( self.ERIinMEM == False ):
            raise RuntimeError ("localintegrals::loc_tei : ERI of the localized orbitals are not stored in memory.")
        return self.activeERI

    # OEIidem means that the OEI is only used to determine the idempotent part of the 1RDM;
    # the correlated part, if it exists, is kept unchanged

    def get_wm_1RDM_from_OEI (self, OEI, nelec=None, loc2wrk=None):

        nelec   = nelec   or self.nelec_idem
        loc2wrk = loc2wrk if np.any (loc2wrk) else self.loc2idem
        nocc    = nelec // 2

        OEI_wrk = represent_operator_in_basis (OEI, loc2wrk)
        oneRDM_wrk = 2 * get_1RDM_from_OEI (OEI_wrk, nocc)
        oneRDM_loc = represent_operator_in_basis (oneRDM_wrk, loc2wrk.T)
        return oneRDM_loc + self.oneRDMcorr_loc

    def get_wm_1RDM_from_scf_on_OEI (self, OEI, nelec=None, loc2wrk=None, ERI_wrk=None, oneRDMguess_loc=None):

        nelec      = nelec   or self.nelec_idem
        loc2wrk    = loc2wrk if np.any (loc2wrk) else self.loc2idem
        ERI_wrk    = ERI_wrk if np.any (ERI_wrk) else self.idemERI
        oneRDM_wrk = represent_operator_in_basis (oneRDMguess_loc, loc2wrk) if np.any (oneRDMguess_loc) else None
        nocc       = nelec // 2

        # DON'T call self.get_wm_1RDM_from_OEIidem here because you need to hold oneRDMcorr_loc frozen until the end of the scf!
        OEI_wrk        = represent_operator_in_basis (OEI, loc2wrk)
        if oneRDM_wrk is None:
            oneRDM_wrk = 2 * get_1RDM_from_OEI (OEI_wrk, nocc)
        if self.ERIinMEM:
            oneRDM_wrk = wm_rhf.solve_ERI(OEI_wrk, ERI_wrk, oneRDM_wrk, nocc, self.num_mf_stab_checks)
        else:
            ao2wrk     = np.dot (self.ao2loc, loc2wrk)
            oneRDM_wrk = wm_rhf.solve_JK (OEI_wrk, self.mol, ao2wrk, oneRDM_wrk, nocc)
        oneRDM_loc     = represent_operator_in_basis (oneRDM_wrk, loc2wrk.T)
        return oneRDM_loc + self.oneRDMcorr_loc

    def setup_wm_core_scf (self, fragments, calcname):

        self.restore_wm_full_scf ()
        oneRDMcorr_loc = sum ((frag.oneRDMas_loc for frag in fragments))
        if np.all (np.isclose (oneRDMcorr_loc, 0)):
            print ("Null correlated 1-RDM; default settings for wm wvfn")
            self.activeJKidem   = self.activeFOCK - self.activeOEI
            self.activeJKcorr   = np.zeros ((self.norbs_tot, self.norbs_tot))
            self.oneRDMcorr_loc = oneRDMcorr_loc
            self.loc2idem       = np.eye (self.norbs_tot)
            self.nelec_idem     = self.nelec_tot
            if ( self.norbs_tot <= 150 ):
                self.idemERI    = self.activeERI
            return

        loc2corr = np.concatenate ([frag.loc2amo for frag in fragments], axis=1)
        loc2idem = get_complementary_states (loc2corr)
        evecs = matrix_eigen_control_options (represent_operator_in_basis (self.loc_oei (), loc2idem), sort_vecs=1, only_nonzero_vals=False)[1]
        loc2idem = np.dot (loc2idem, evecs)

        # I want to alter the outputs of self.loc_oei (), self.loc_rhf_fock (), and the get_wm_1RDM_etc () functions.
        # self.loc_oei ()      = P_idem * (activeOEI + JKcorr) * P_idem
        # self.loc_rhf_fock () = P_idem * (activeOEI + JKcorr + JKidem) * P_idem
        # The get_wm_1RDM_etc () functions will need to add oneRDMcorr_loc to their final return value
        # The chemical potential is so that identically zero eigenvalues from the projection into the idem space don't get confused
        # with numerically-zero eigenvalues in the idem space: all occupied orbitals must have negative energy
        
        # Make true output 1RDM from fragments to use as guess for wm mcscf calculation
        oneRDMguess_loc = np.zeros_like (oneRDMcorr_loc)
        for f in itertools.product (fragments, fragments):
            loc2frag         = [i.loc2frag for i in f] 
            oneRDMguess_loc += sum ((0.5 * project_operator_into_subspace (i.oneRDM_loc, *loc2frag) for i in f))

        nelec_corr     = np.trace (oneRDMcorr_loc)
        if is_close_to_integer (nelec_corr, params.num_zero_atol) == False:
            raise ValueError ("nelec_corr not an integer! {}".format (nelec_corr))
        nelec_idem     = int (round (self.nelec_tot - nelec_corr))
        JKcorr         = self.loc_rhf_jk_bis (oneRDMcorr_loc)
        idemERI        = None
        if self.ERIinMEM:
            norbs      = loc2idem.shape[1]
            ao2idem    = np.dot (self.ao2loc, loc2idem)
            idemERI    = ao2mo.incore.full(ao2mo.restore(8, self.activeERI, self.norbs_tot), loc2idem, compact=False).reshape(norbs, norbs, norbs, norbs)
        oneRDMidem_loc = self.get_wm_1RDM_from_scf_on_OEI (self.loc_oei () + JKcorr, nelec=nelec_idem, loc2wrk=loc2idem, ERI_wrk=idemERI, oneRDMguess_loc=oneRDMguess_loc)
        JKidem         = self.loc_rhf_jk_bis (oneRDMidem_loc)
        print ("trace of oneRDMcorr_loc = {}".format (np.trace (oneRDMcorr_loc)))
        print ("trace of oneRDMidem_loc = {}".format (np.trace (oneRDMidem_loc)))
        print ("trace of oneRDM_loc in corr basis = {}".format (np.trace (represent_operator_in_basis (oneRDMcorr_loc + oneRDMidem_loc, orthonormalize_a_basis (loc2corr)))))
        svals = get_overlapping_states (loc2idem, loc2corr)[2]
        print ("trace of <idem|corr|idem> = {}".format (np.sum (svals * svals)))
        print (loc2corr.shape)
        print (loc2idem.shape)

        ########################################################################################################        
        self.activeJKidem   = JKidem
        self.activeJKcorr   = JKcorr
        self.oneRDMcorr_loc = oneRDMcorr_loc
        self.loc2idem       = loc2idem
        self.nelec_idem     = nelec_idem
        self.idemERI        = idemERI
        ########################################################################################################

        # Analysis: molden and total energy
        print ("Analyzing LASSCF trial wave function")
        oei = self.activeOEI + (JKcorr + JKidem) / 2
        fock = self.activeOEI + JKcorr + JKidem
        oneRDM = oneRDMidem_loc + oneRDMcorr_loc
        E = self.activeCONST + np.tensordot (oei, oneRDM, axes=2)
        for frag in fragments:
            if frag.norbs_as > 0:
                V  = self.dmet_tei (frag.loc2amo)
                L  = frag.twoCDMimp_amo
                E += np.tensordot (V, L, axes=4) / 2
        print ("LASSCF trial wave function total energy: {:.6f}".format (E))
        fock_idem = represent_operator_in_basis (fock, loc2idem)
        oneRDM_corr = represent_operator_in_basis (oneRDM, loc2corr)
        idem_evecs = matrix_eigen_control_options (fock_idem, sort_vecs=1, only_nonzero_vals=False)[1]
        corr_evecs =  matrix_eigen_control_options (oneRDM_corr, sort_vecs=-1, only_nonzero_vals=False)[1]
        loc2molden = np.append (np.dot (loc2idem, idem_evecs), np.dot (loc2corr, corr_evecs), axis=1)
        wm_ene = np.einsum ('ip,ij,jp->p', loc2molden, fock, loc2molden)
        wm_occ = np.einsum ('ip,ij,jp->p', loc2molden, oneRDM, loc2molden)
        ao2molden = np.dot (self.ao2loc, loc2molden)
        molden.from_mo (self.mol, calcname + '_trial_wvfn.molden', ao2molden, occ=wm_occ, ene=wm_ene)

    def restore_wm_full_scf (self):
        self.activeJKidem   = self.activeFOCK - self.activeOEI
        self.activeJKcorr   = np.zeros ((self.norbs_tot, self.norbs_tot), dtype=self.activeOEI.dtype)
        self.oneRDMcorr_loc = np.zeros ((self.norbs_tot, self.norbs_tot), dtype=self.activeOEI.dtype)
        self.loc2idem       = np.eye (self.norbs_tot, dtype=self.activeOEI.dtype)
        self.nelec_idem     = self.nelec_tot
        self.idemERI        = self.activeERI

    def dmet_oei( self, loc2dmet, numActive ):
    
        OEIdmet  = np.dot( np.dot( loc2dmet[:,:numActive].T, self.activeOEI ), loc2dmet[:,:numActive] )
        return symmetrize_tensor (OEIdmet)
        
    def dmet_fock( self, loc2dmet, numActive, coreDMloc ):
    
        FOCKdmet  = np.dot( np.dot( loc2dmet[:,:numActive].T, self.loc_rhf_fock_bis( coreDMloc ) ), loc2dmet[:,:numActive] )
        return symmetrize_tensor (FOCKdmet)
        
    def dmet_init_guess_rhf( self, loc2dmet, numActive, numPairs, norbs_frag, chempot_imp ):
    
        Fock_small = np.dot( np.dot( loc2dmet[:,:numActive].T, self.loc_rhf_fock ()), loc2dmet[:,:numActive] )
        if (chempot_imp != 0.0):
            Fock_small[np.diag_indices(norbs_frag)] -= chempot_imp
        eigvals, eigvecs = np.linalg.eigh( Fock_small )
        eigvecs = eigvecs[ :, eigvals.argsort() ]
        DMguess = 2 * np.dot( eigvecs[ :, :numPairs ], eigvecs[ :, :numPairs ].T )
        return DMguess
        
    def dmet_tei( self, loc2dmet, numAct=None ):

        numAct = loc2dmet.shape[1] if numAct==None else numAct
        loc2imp = loc2dmet[:,:numAct]
        TEI = symmetrize_tensor (self.general_tei ([loc2imp for i in range(4)]))
        return TEI

    def dmet_const (self, loc2dmet, norbs_imp, oneRDMfroz_loc):
        norbs_core = self.norbs_tot - norbs_imp
        if norbs_core == 0:
            return 0.0
        loc2core = loc2dmet[:,norbs_imp:]
        GAMMA = represent_operator_in_basis (oneRDMfroz_loc, loc2core)
        OEI = self.dmet_oei (loc2core, norbs_core)
        FOCK = self.dmet_fock (loc2core, norbs_core, oneRDMfroz_loc)
        return 0.5 * np.einsum ('ij,ij->', GAMMA, OEI + FOCK)

    def general_tei (self, loc2bas_list):
        norbs = [loc2bas.shape[1] for loc2bas in loc2bas_list]

        if self.ERIinMEM:
            TEI = ao2mo.incore.general(ao2mo.restore(8, self.activeERI, self.norbs_tot), loc2bas_list, compact=False).reshape (*norbs)
        else:
            a2b_list = [np.dot (self.ao2loc, l2b) for l2b in loc2bas_list]
            TEI  = ao2mo.outcore.general_iofree(self.mol, a2b_list, compact=False).reshape (*norbs)

        return TEI

    def compare_basis_to_loc (self, loc2bas, frags, nlead=3, quiet=True):
        nfrags = len (frags)
        norbs_tot, norbs_bas = loc2bas.shape
        if norbs_bas == 0:
            return np.zeros (nfrags), loc2bas
        my_dtype  = sum ([[('weight{0}'.format (i), 'f8'), ('frag{0}'.format (i), 'U3')] for i in range (nfrags)], [])
        my_dtype += sum ([[('coeff{0}'.format (i), 'f8'), ('coord{0}'.format (i), 'U9')] for i in range (nlead)],  [])
        analysis = np.array ([ sum (((0, '-') for j in range (len (my_dtype) // 2)), tuple()) for i in range (norbs_bas) ], dtype=my_dtype)
        bas_weights   = np.asarray ([np.diag (represent_operator_in_basis (np.diag (f.is_frag_orb.astype (int)), loc2bas)) for f in frags]).T
        bas_frags_idx = np.argsort (bas_weights, axis=1)[:,::-1]
        bas_weights   = np.sort    (bas_weights, axis=1)[:,::-1]
        for j in range (nfrags):
            analysis['weight{0}'.format (j)] = bas_weights[:,j]
            analysis['frag{0}'.format (j)] = [frags[i].frag_name for i in bas_frags_idx[:,j]]
    
        def find_frag_fragorb (loc_orbs):
            thefrag     = [np.where ([f.is_frag_orb[i] for f in frags])[0][0] for i in loc_orbs]
            thefragorb  = [np.where (frags[i].frag_orb_list == j)[0][0] for i, j in zip (thefrag, loc_orbs)]
            thefragname = [frags[i].frag_name for i in thefrag]
            thestring = ['{:d}:{:s}'.format (idx, name) for name, idx in zip (thefragname, thefragorb)]
            return thestring
    
        weights_idx0 = np.argsort (np.absolute (loc2bas), axis=0)[:-nlead-1:-1,:]
        weights_idx1 = np.array ([range (norbs_bas) for i in range (nlead)])
        leading_coeffs = loc2bas[weights_idx0,weights_idx1].T
        overall_idx = np.argsort (weights_idx0[0,:])
        for j in range (nlead):
            analysis['coeff{0}'.format (j)] = leading_coeffs[:,j]
            analysis['coord{0}'.format (j)] = find_frag_fragorb (weights_idx0[j,:])
        analysis = analysis[overall_idx]
    
        if quiet == False:
            format_str = ' '.join (['{:' + str (len (name)) + 's}' for name in analysis.dtype.names])
            print (format_str.format (*analysis.dtype.names))
            format_str  = ' '.join (sum([['{:'  + str (len (analysis.dtype.names[2*i]))     + '.2f}',
                                          '{:>' + str (len (analysis.dtype.names[(2*i)+1])) + 's}']
                                        for i in range (nfrags + nlead)], []))
            for i in range (norbs_bas):
                print (format_str.format (*analysis[i]))
            print ("Worst fragment localization: {:.2f}".format (np.amin (analysis['weight0'])))
    
        return loc2bas[:,overall_idx], np.array ([np.count_nonzero (analysis['frag0'] == f.frag_name) for f in frags])
        


    def relocalize_states (self, loc2bas, fragments, oneRDM_loc, natorb=False, canonicalize=False):
        '''Do Boys localization on a subspace and assign resulting states to the various fragments using projection operators.
           Optionally diagonalize either the fock or the density matrix inside each subspace. Canonicalize overrides natorb'''

        fock_loc = self.loc_rhf_fock_bis (oneRDM_loc)
        ao2bas = boys.Boys (self.mol, np.dot (self.ao2loc, loc2bas)).kernel ()
        loc2bas = reduce (np.dot, [self.ao2loc.conjugate ().T, self.ao_ovlp, ao2bas])
        
        weights = np.asarray ([np.einsum ('ip,ip->p', loc2bas[f.frag_orb_list,:].conjugate (), loc2bas[f.frag_orb_list,:]) for f in fragments])
        frag_assignments = np.argmax (weights, axis=0)

        loc2bas_assigned = []        
        for idx, frag in enumerate (fragments):
            pick_orbs = (frag_assignments == idx)
            norbs = np.count_nonzero (pick_orbs)
            print ("{} states found for fragment {}".format (norbs, frag.frag_name))
            loc2pick = loc2bas[:,pick_orbs]
            if canonicalize and norbs:
                f = represent_operator_in_basis (fock_loc, loc2pick)
                evals, evecs = matrix_eigen_control_options (f, sort_vecs=1, only_nonzero_vals=False)
                loc2pick = np.dot (loc2pick, evecs)
            elif natorb and norbs:
                f = represent_operator_in_basis (oneRDM_loc, loc2pick)
                evals, evecs = matrix_eigen_control_options (f, sort_vecs=-1, only_nonzero_vals=False)
                loc2pick = np.dot (loc2pick, evecs)
            loc2bas_assigned.append (loc2pick)
        return loc2bas_assigned


