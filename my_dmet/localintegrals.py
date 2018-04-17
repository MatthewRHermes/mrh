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
import . import rhf as wm_rhf
from . import iao_helper
import numpy as np
from mrh.util.basis import represent_operator_in_basis
from mrh.util.la import matrix_eigen_control_options

class localintegrals:

    def __init__( self, the_mf, active_orbs, localizationtype, ao_rotation=None, use_full_hessian=True, localization_threshold=1e-6 ):

        assert (( localizationtype == 'meta_lowdin' ) or ( localizationtype == 'boys' ) or ( localizationtype == 'lowdin' ) or ( localizationtype == 'iao' ))
        self.num_zero_atol = 1.0e-8
        
        # Information on the full HF problem
        self.mol        = the_mf.mol
        self.fullovlpao = the_mf.get_ovlp
        self.fullEhf    = the_mf.e_tot
        self.fullDMao   = np.dot(np.dot( the_mf.mo_coeff, np.diag( the_mf.mo_occ )), the_mf.mo_coeff.T )
        self.fullJKao   = scf.hf.get_veff( self.mol, self.fullDMao, 0, 0, 1 ) #Last 3 numbers: dm_last, vhf_last, hermi
        self.fullFOCKao = self.mol.intor('cint1e_kin_sph') + self.mol.intor('cint1e_nuc_sph') + self.fullJKao
        
        # Active space information
        self._which   = localizationtype
        self.active   = np.zeros( [ self.mol.nao_nr() ], dtype=int )
        self.active[ active_orbs ] = 1
        self.Norbs    = np.sum( self.active ) # Number of active space orbitals
        self.Nelec    = int(np.rint( self.mol.nelectron - np.sum( the_mf.mo_occ[ self.active==0 ] ))) # Total number of electrons minus frozen part
        
        # Localize the orbitals
        if (( self._which == 'meta_lowdin' ) or ( self._which == 'boys' )):
            if ( self._which == 'meta_lowdin' ):
                assert( self.Norbs == self.mol.nao_nr() ) # Full active space required
            if ( self._which == 'boys' ):
                self.ao2loc = the_mf.mo_coeff[ : , self.active==1 ]
            if ( self.Norbs == self.mol.nao_nr() ): # If you want the full active, do meta-Lowdin
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
            assert( self.Norbs == self.mol.nao_nr() ) # Full active space required
            ovlp = self.mol.intor('cint1e_ovlp_sph')
            ovlp_eigs, ovlp_vecs = np.linalg.eigh( ovlp )
            assert ( np.linalg.norm( np.dot( np.dot( ovlp_vecs, np.diag( ovlp_eigs ) ), ovlp_vecs.T ) - ovlp ) < 1e-10 )
            self.ao2loc = np.dot( np.dot( ovlp_vecs, np.diag( np.power( ovlp_eigs, -0.5 ) ) ), ovlp_vecs.T )
            self.TI_OK  = False # Check yourself if OK, then overwrite
        if ( self._which == 'iao' ):
            assert( self.Norbs == self.mol.nao_nr() ) # Full active space assumed
            self.ao2loc = iao_helper.localize_iao( self.mol, the_mf )
            if ( ao_rotation != None ):
                self.ao2loc = np.dot( self.ao2loc, ao_rotation.T )
            self.TI_OK = False # Check yourself if OK, then overwrite
            #self.molden( 'dump.molden' ) # Debugging mode
        assert( self.loc_ortho() < 1e-8 )
        
        # Effective Hamiltonian due to frozen part
        self.frozenDMmo  = np.array( the_mf.mo_occ, copy=True )
        self.frozenDMmo[ self.active==1 ] = 0 # Only the frozen MO occupancies nonzero
        self.frozenDMao  = np.dot(np.dot( the_mf.mo_coeff, np.diag( self.frozenDMmo )), the_mf.mo_coeff.T )
        self.frozenJKao  = scf.hf.get_veff( self.mol, self.frozenDMao, 0, 0, 1 ) #Last 3 numbers: dm_last, vhf_last, hermi
        self.frozenOEIao = self.fullFOCKao - self.fullJKao + self.frozenJKao
        
        # Localized OEI and ERI
        self.activeCONST = self.mol.energy_nuc() + np.einsum( 'ij,ij->', self.frozenOEIao - 0.5*self.frozenJKao, self.frozenDMao )
        self.activeOEI   = np.dot( np.dot( self.ao2loc.T, self.frozenOEIao ), self.ao2loc )
        self.activeFOCK  = np.dot( np.dot( self.ao2loc.T, self.fullFOCKao  ), self.ao2loc )
        if ( self.Norbs <= 150 ):
            self.ERIinMEM  = True
            self.activeERI = ao2mo.outcore.full_iofree( self.mol, self.ao2loc, compact=False ).reshape(self.Norbs, self.Norbs, self.Norbs, self.Norbs)
        else:
            self.ERIinMEM  = False
            self.activeERI = None
        
        self.loc2wmc = np.eye (self.Norbs)
        self.nelec_wmc = self.Nelec
        self.TEI_wmc = None
        self.ao2wmc = None
        if self.ERIinMEM:
            self.TEI_wmc = self.activeERI
        else:
            self.ao2wmc = self.ao2loc
        #self.debug_matrixelements()
        
    def molden( self, filename ):
    
        with open( filename, 'w' ) as thefile:
            molden.header( self.mol, thefile )
            molden.orbital_coeff( self.mol, thefile, self.ao2loc )
            
    def loc_ortho( self ):
    
#        ShouldBeI = np.dot( np.dot( self.ao2loc.T , self.mol.intor('cint1e_ovlp_sph') ) , self.ao2loc )
        ShouldBeI = np.dot( np.dot( self.ao2loc.T , self.fullovlpao () ) , self.ao2loc )
        return np.linalg.norm( ShouldBeI - np.eye( ShouldBeI.shape[0] ) )
        
    def debug_matrixelements( self ):
    
        eigvals, eigvecs = np.linalg.eigh( self.activeFOCK )
        eigvecs = eigvecs[ :, eigvals.argsort() ]
        assert( self.Nelec % 2 == 0 )
        numPairs = self.Nelec // 2
        DMguess = 2 * np.dot( eigvecs[ :, :numPairs ], eigvecs[ :, :numPairs ].T )
        if ( self.ERIinMEM == True ):
            DMloc = wm_rhf.solve_ERI( self.activeOEI, self.activeERI, DMguess, numPairs )
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
        
        return self.activeOEI
        
    def loc_rhf_fock( self ):

        return project_operator_into_basis (self.activeFOCK, self.loc2wmc)
        
    def loc_rhf_jk_bis( self, DMloc ):
    
        if ( self.ERIinMEM == False ):
            DM_ao = np.dot( np.dot( self.ao2loc, DMloc ), self.ao2loc.T )
            JK_ao = scf.hf.get_veff( self.mol, DM_ao, 0, 0, 1 ) #Last 3 numbers: dm_last, vhf_last, hermi
            JK_loc = np.dot( np.dot( self.ao2loc.T, JK_ao ), self.ao2loc )
        else:
            JK_loc = np.einsum( 'ijkl,ij->kl', self.activeERI, DMloc ) - 0.5 * np.einsum( 'ijkl,ik->jl', self.activeERI, DMloc )
        return JK_loc

    def loc_rhf_fock_bis( self, DMloc ):
    
        return self.activeOEI + self.loc_rhf_jk_bis (DMloc)

    def loc_tei( self ):
    
        if ( self.ERIinMEM == False ):
            raise RuntimeError ("localintegrals::loc_tei : ERI of the localized orbitals are not stored in memory.")
        return self.activeERI

    def do_wm_scf (self, custom_OEI_loc):

        npairs_wmc = self.nelec_wmc // 2
        OEI_wmc = represent_operator_in_basis (custom_OEI_loc, self.loc2wmc)
        _, wmc2mo = matrix_eigen_control_options (OEI_wmc, sort_vecs=True) # Sorts them ~highest-to-lowest~, so I have to reverse it
        wmc2mo = wmc2mo[:,::-1]
        wmc2mo = wmc2mo[:,:npairs_wmc]
        oneRDM_wmc = 2 * np.dot (wmc2mo, wmc2mo.T)
        if self.ERIinMEM:
            oneRDM_wmc = wm_rhf.solve_ERI (OEI_wmc, self.TEI_wmc, oneRDM_wmc, npairs_wmc)
        else:
            oneRDM_wmc = wm_rhf.solve_JK (OEI_wmc, self.mol, self.ao2wmc, oneRDM_wmc, npairs_wmc)
        return represent_operator_in_basis (oneRDM_wmc, self.loc2wmc.T)        

    def setup_wmc (self, oneRDMwma_loc):

        nelec_wma = np.trace (oneRDMwma_loc)
        if abs ((2*round (nelec_wma/2)) - nelec_wma) > self.num_zero_atol:
            raise RuntimeError ("oneRDMwma_loc has non-even-integer number electrons ({0})".format (nelec_wma))
        self.nelec_wmc = self.Nelec - round (self.nelec_wma) # important line
        GOCK_loc = self.loc_rhf_fock_bis (oneRDMwma_loc)
        self.loc2wmc = wm_rhf.get_unfrozen_states (oneRDMwma_loc) # important line
        oneRDMwmc_loc = self.do_wm_scf (GOCK_loc)
        self.activeFOCK = self.loc_rhf_fock_bis (oneRDMwma_loc + oneRDMwmc_loc) # important line
        if self.ERIinMEM:
            self.TEI_wmc = self.dmet_tei (self.loc2wmc, self.loc2wmc.shape[1])
        else:
            self.ao2wmc = np.dot (self.ao2loc, self.loc2wmc)

    def restore_original_wm_scf (self):
        self.nelec_wmc   = self.Nelec
        self.loc2wmc     = np.eye (self.Norbs)
        self.activeFOCK  = np.dot( np.dot( self.ao2loc.T, self.fullFOCKao  ), self.ao2loc )
        if self.ERIinMEM:
            self.TEI_wmc = self.activeERI
        else:
            self.ao2wmc = self.ao2loc

    def dmet_oei( self, loc2dmet, numActive ):
    
        OEIdmet = np.dot( np.dot( loc2dmet[:,:numActive].T, self.activeOEI ), loc2dmet[:,:numActive] )
        return OEIdmet
        
    def dmet_fock( self, loc2dmet, numActive, coreDMloc ):
    
        FOCKdmet = np.dot( np.dot( loc2dmet[:,:numActive].T, self.loc_rhf_fock_bis( coreDMloc ) ), loc2dmet[:,:numActive] )
        return FOCKdmet
        
    def dmet_init_guess_rhf( self, loc2dmet, numActive, numPairs, norbs_frag, chempot_imp ):
    
        Fock_small = np.dot( np.dot( loc2dmet[:,:numActive].T, self.activeFOCK ), loc2dmet[:,:numActive] )
        if (chempot_imp != 0.0):
            Fock_small[np.diag_indices(norbs_frag)] -= chempot_imp
        eigvals, eigvecs = np.linalg.eigh( Fock_small )
        eigvecs = eigvecs[ :, eigvals.argsort() ]
        DMguess = 2 * np.dot( eigvecs[ :, :numPairs ], eigvecs[ :, :numPairs ].T )
        return DMguess
        
    def dmet_tei( self, loc2dmet, numAct ):
    
        if ( self.ERIinMEM == False ):
            transfo = np.dot( self.ao2loc, loc2dmet[:,:numAct] )
            TEIdmet = ao2mo.outcore.full_iofree(self.mol, transfo, compact=False).reshape(numAct, numAct, numAct, numAct)
        else:
            TEIdmet = ao2mo.incore.full(ao2mo.restore(8, self.activeERI, self.Norbs), loc2dmet[:,:numAct], compact=False).reshape(numAct, numAct, numAct, numAct)
        return TEIdmet
        
    def dmet_electronic_const (self, loc2dmet, norbs_imp, oneRDMwm_loc):

        norbs_tot=self.mol.nao_nr ()
        norbs_core=norbs_tot - norbs_imp
        loc2core = loc2dmet[:,::-1] 
        GAMMA = represent_operator_in_basis (oneRDMwm_loc, loc2core[:,:norbs_core])
        OEI = self.dmet_oei (loc2core, norbs_core)
        FOCK = self.dmet_fock (loc2core, norbs_core, oneRDMwm_loc)
        return 0.5 * np.einsum ('ij,ij->', GAMMA, OEI + FOCK)

 
