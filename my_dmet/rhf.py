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

import numpy as np
import scipy.sparse.linalg
#import qcdmet_paths
from pyscf import gto, scf, ao2mo
from mrh.util.basis import get_complementary_states
from mrh.util.la import matrix_eigen_control_options

def solve_ERI( OEI, TEI, oneRDMguess_loc, numPairs, num_mf_stab_checks):

    mol = gto.Mole()
    mol.build(verbose=0)
    mol.atom.append(('C', (0, 0, 0)))
    mol.nelectron = 2 * numPairs

    L = OEI.shape[0]
    mf = scf.RHF( mol )
    mf.get_hcore = lambda *args: OEI
    mf.get_ovlp = lambda *args: np.eye( L )
    mf._eri = ao2mo.restore(8, TEI, L)
    mf.verbose=0
    mf.scf( oneRDMguess_loc )
    #oneRDM_loc = np.dot(np.dot( mf.mo_coeff, np.diag( mf.mo_occ )), mf.mo_coeff.T )
    oneRDM_loc = mf.make_rdm1 ()
    if ( mf.converged == False ):
        mf.newton ().kernel ( oneRDM_loc )
        oneRDM_loc = mf.make_rdm1 () #np.dot(np.dot( mf.mo_coeff, np.diag( mf.mo_occ )), mf.mo_coeff.T )

    # Instability check and repeat
    for i in range (num_mf_stab_checks):
        mf.mo_coeff = mf.stability ()[0]
        oneRDMguess_loc = mf.make_rdm1 ()
        mf = scf.RHF( mol )
        mf.get_hcore = lambda *args: OEI
        mf.get_ovlp = lambda *args: np.eye( L )
        mf._eri = ao2mo.restore(8, TEI, L)
        mf.verbose=0
        mf.scf( oneRDMguess_loc )
        oneRDM_loc = mf.make_rdm1 ()
        if ( mf.converged == False ):
            mf.newton ().kernel ( oneRDM_loc )
            oneRDM_loc = mf.make_rdm1 () #np.dot(np.dot( mf.mo_coeff, np.diag( mf.mo_occ )), mf.mo_coeff.T )
    
    return oneRDM_loc
    
def wrap_my_jk( mol_orig, ao2basis ): # mol_orig works in ao

    #get_jk(mol, dm, hermi=1, vhfopt=None)
    def my_jk( mol, dm, hermi=1, vhfopt=None ): # mol works in basis, dm is in basis
    
        dm_ao        = np.dot( np.dot( ao2basis, dm ), ao2basis.T )
        vj_ao, vk_ao = scf.hf.get_jk( mol_orig, dm_ao, hermi, vhfopt )
        vj_basis     = np.dot( np.dot( ao2basis.T, vj_ao ), ao2basis )
        vk_basis     = np.dot( np.dot( ao2basis.T, vk_ao ), ao2basis )
        return vj_basis, vk_basis
    
    return my_jk

def wrap_my_veff( mol_orig, ao2basis ): # mol_orig works in ao

    #get_veff(mol, dm, dm_last=0, vhf_last=0, hermi=1, vhfopt=None)
    def my_veff( mol, dm, dm_last=0, vhf_last=0, hermi=1, vhfopt=None ): # mol works in basis, dm is in basis
        
        ddm_basis    = np.array(dm, copy=False) - np.array(dm_last, copy=False)
        ddm_ao       = np.dot( np.dot( ao2basis, ddm_basis ), ao2basis.T )
        vj_ao, vk_ao = scf.hf.get_jk( mol_orig, ddm_ao, hermi, vhfopt )
        veff_ao      = vj_ao - 0.5 * vk_ao
        veff_basis   = np.dot( np.dot( ao2basis.T, veff_ao ), ao2basis ) + np.array( vhf_last, copy=False )
        return veff_basis
        
    return my_veff

def solve_JK( OEI, mol_orig, ao2basis, oneRDMguess_loc, numPairs):

    mol = gto.Mole()
    mol.build(verbose=0)
    mol.atom.append(('C', (0, 0, 0)))
    mol.nelectron = 2 * numPairs

    L = OEI.shape[0]
    mf = scf.RHF( mol )
    mf.get_hcore = lambda *args: OEI
    mf.get_ovlp = lambda *args: np.eye( L )
    mf._eri = None
    mf.get_jk   = wrap_my_jk(   mol_orig, ao2basis )
    mf.get_veff = wrap_my_veff( mol_orig, ao2basis )
    mf.max_cycle = 500
    mf.damp_factor = 0.33
    
    mf.scf( oneRDMguess_loc )
    oneRDM_loc = np.dot(np.dot( mf.mo_coeff, np.diag( mf.mo_occ )), mf.mo_coeff.T )
    return oneRDM_loc
    
def get_unfrozen_states (oneRDMfroz_loc):
    _, loc2froz = matrix_eigen_control_options (oneRDMfroz_loc, only_nonzero_vals=True)
    if loc2froz.shape[1] == loc2froz.shape[0]:
        raise RuntimeError ("No unfrozen states: eigenbasis of oneRDMfroz_loc is complete!")
    return get_complementary_states (loc2froz)



