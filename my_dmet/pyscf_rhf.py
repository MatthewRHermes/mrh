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
#import qcdmet_paths
from . import localintegrals
from pyscf import ao2mo, gto, scf
from pyscf.tools import rhf_newtonraphson

#def solve( CONST, OEI, FOCK, TEI, frag.norbs_imp, frag.nelec_imp, frag.norbs_frag, guess_1RDM, chempot_frag=0.0 ):
def solve (frag, guess_1RDM, chempot_frag=0.0):

    # Augment the FOCK operator with the chemical potential
    FOCKcopy = frag.impham_FOCK.copy()
    if (chempot_frag != 0.0):
        for orb in range(frag.norbs_frag):
            FOCKcopy[ orb, orb ] -= chempot_frag
    
    # Get the RHF solution
    mol = gto.Mole()
    mol.build( verbose=0 )
    mol.atom.append(('C', (0, 0, 0)))
    mol.nelectron = frag.nelec_imp
    mol.incore_anyway = True
    mf = scf.RHF( mol )
    mf.get_hcore = lambda *args: FOCKcopy
    mf.get_ovlp = lambda *args: np.eye( frag.norbs_imp )
    mf._eri = ao2mo.restore(8, frag.impham_TEI, frag.norbs_imp)
    mf.scf( guess_1RDM )
    DMloc = np.dot(np.dot( mf.mo_coeff, np.diag( mf.mo_occ )), mf.mo_coeff.T )
    if ( mf.converged == False ):
        mf = rhf_newtonraphson.solve( mf, dm_guess=DMloc )
        DMloc = np.dot(np.dot( mf.mo_coeff, np.diag( mf.mo_occ )), mf.mo_coeff.T )
    
    frag.E_imp = mf.e_tot
    frag.oneRDM_imp = mf.make_rdm1()
    JK   = mf.get_veff(None, dm=frag.oneRDM_imp)
 
    # To calculate the impurity energy, rescale the JK matrix with a factor 0.5 to avoid double counting: 0.5 * ( OEI + FOCK ) = OEI + 0.5 * JK
    frag.E_frag =  0.25 * np.einsum('ji,ij->', frag.oneRDM_imp[:,:frag.norbs_frag], frag.impham_FOCK[:frag.norbs_frag,:] + frag.impham_OEI[:frag.norbs_frag,:]) \
                 + 0.25 * np.einsum('ji,ij->', frag.oneRDM_imp[:frag.norbs_frag,:], frag.impham_FOCK[:,:frag.norbs_frag] + frag.impham_OEI[:,:frag.norbs_frag]) \
                 + 0.25 * np.einsum('ji,ij->', frag.oneRDM_imp[:,:frag.norbs_frag], JK[:frag.norbs_frag,:]) \
                 + 0.25 * np.einsum('ji,ij->', frag.oneRDM_imp[:frag.norbs_frag,:], JK[:,:frag.norbs_frag])
    
    return None

