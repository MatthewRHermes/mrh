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
from mrh.my_dmet import localintegrals
from pyscf import ao2mo, gto, scf
from pyscf.tools import rhf_newtonraphson
from mrh.util.basis import represent_operator_in_basis, project_operator_into_subspace
from mrh.util.tensors import symmetrize_tensor

#def solve( CONST, OEI, FOCK, TEI, frag.norbs_imp, frag.nelec_imp, frag.norbs_frag, guess_1RDM, chempot_frag=0.0 ):
def solve (frag, guess_1RDM, chempot_imp):

    # Augment OEI with the chemical potential
    OEI = frag.impham_OEI - chempot_imp

    # Testing: load hamiltonian from working copy
    if not hasattr (frag, 'loaded'):
        guess_1RDM_wrking = np.load (frag.frag_name + '_1rdm.npy')
        OEI_wrking = np.load (frag.frag_name + '_oei.npy')
        TEI_wrking = np.load (frag.frag_name + '_tei.npy')
        frag.loaded = True
        print ("guess_1RDM versus guess_1RDM_wrking: {0}".format (np.linalg.norm (guess_1RDM - guess_1RDM_wrking)))
        print ("OEI versus OEI_wrking: {0}".format (np.linalg.norm (OEI - OEI_wrking)))
        print ("TEI versus TEI_wrking: {0}".format (np.linalg.norm (frag.impham_TEI - TEI_wrking)))
        molt = gto.Mole()
        molt.build( verbose=0 )
        molt.atom.append(('C', (0, 0, 0)))
        molt.nelectron = frag.nelec_imp
        molt.incore_anyway = True
        mft = scf.RHF( molt )
        mft.get_hcore = lambda *args: OEI_wrking
        mft.get_ovlp = lambda *args: np.eye( frag.norbs_imp )
        mft._eri = ao2mo.restore(8, TEI_wrking, frag.norbs_imp)
        mft.scf(guess_1RDM_wrking)
        print ("Ham_wrking E_scf = {0}".format (mft.e_tot))

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
    oneRDMimp_imp = mf.make_rdm1()    
    print ("This branch E_scf = {0}".format (mf.e_tot))

    frag.oneRDM_loc = symmetrize_tensor (frag.oneRDMfroz_loc + represent_operator_in_basis (oneRDMimp_imp, frag.imp2loc))
    frag.twoCDM_imp = np.zeros_like (frag.impham_TEI)
    frag.E_imp      = frag.impham_CONST + mf.e_tot + np.einsum ('ab,ab->', oneRDMimp_imp, chempot_imp)
    frag.loc2mo     = np.dot (frag.loc2imp, mf.mo_coeff)
    
    return None

