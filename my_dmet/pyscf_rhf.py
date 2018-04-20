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
from mrh.util.basis import represent_operator_in_basis
from mrh.util.tensors import symmetrize_tensor

#def solve( CONST, OEI, FOCK, TEI, frag.norbs_imp, frag.nelec_imp, frag.norbs_frag, guess_1RDM, chempot_frag=0.0 ):
def solve (frag, guess_1RDM, chempot_frag=0.0):

    # Augment OEI operator with the chemical potential
    chempot = represent_operator_in_basis (chempot_frag * np.eye (frag.norbs_frag), frag.frag2imp)
    OEI = frag.impham_OEI - chempot
    
    # Get the RHF solution
    mol = gto.Mole()
    mol.build( verbose=0 )
    mol.atom.append(('C', (0, 0, 0)))
    mol.nelectron = frag.nelec_imp
    mol.incore_anyway = True
    mf = scf.RHF( mol )
    mf.get_hcore = lambda *args: OEI
    mf.get_ovlp = lambda *args: np.eye( frag.norbs_imp )
    mf._eri = ao2mo.restore(8, frag.impham_TEI, frag.norbs_imp)
    mf.scf( guess_1RDM )
    DMloc = np.dot(np.dot( mf.mo_coeff, np.diag( mf.mo_occ )), mf.mo_coeff.T )
    if ( mf.converged == False ):
        mf = rhf_newtonraphson.solve( mf, dm_guess=DMloc )
    
    frag.E_imp       = frag.impham_CONST + mf.e_tot
    frag.oneRDM_imp  = mf.make_rdm1()
    frag.twoRDMR_imp = np.zeros ((frag.norbs_imp, frag.norbs_imp, frag.norbs_imp, frag.norbs_imp))
    
    return None

