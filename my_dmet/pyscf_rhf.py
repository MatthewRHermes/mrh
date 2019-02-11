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
import time
#import qcdmet_paths
from mrh.my_dmet import localintegrals
from pyscf import ao2mo, gto, scf
from mrh.util.basis import represent_operator_in_basis, project_operator_into_subspace, measure_basis_olap
from mrh.util.tensors import symmetrize_tensor
from functools import reduce

#def solve( CONST, OEI, FOCK, TEI, frag.norbs_imp, frag.nelec_imp, frag.norbs_frag, guess_1RDM, chempot_frag=0.0 ):
def solve (frag, guess_1RDM, chempot_imp):

    t_start = time.time ()

    # Augment OEI with the chemical potential
    OEI = frag.impham_OEI - chempot_imp

    # Get the RHF solution
    mol = gto.Mole()
    mol.spin = int (round (2 * frag.target_MS))
    mol.verbose = 0 if frag.mol_output is None else 4
    mol.output = frag.mol_output
    mol.build ()
    mol.atom.append(('C', (0, 0, 0)))
    mol.nelectron = frag.nelec_imp
    mol.incore_anyway = True
    mf = scf.RHF( mol )
    mf.get_hcore = lambda *args: OEI
    mf.get_ovlp = lambda *args: np.eye( frag.norbs_imp )
    if frag.quasidirect:
        mf.get_jk = frag.impham_get_jk 
    else:
        mf._eri = ao2mo.restore(8, frag.impham_TEI, frag.norbs_imp)
    mf.__dict__.update (frag.mf_attr)
    mf.scf( guess_1RDM )
    if ( mf.converged == False ):
        mf = mf.newton ()
        mf.kernel ()

    # Instability check and repeat
    for i in range (frag.num_mf_stab_checks):
        new_mo = mf.stability ()[0]
        guess_1RDM = reduce (np.dot, (new_mo, np.diag (mf.mo_occ), new_mo.conjugate ().T))
        mf = scf.RHF( mol )
        mf.get_hcore = lambda *args: OEI
        mf.get_ovlp = lambda *args: np.eye( frag.norbs_imp )
        if frag.quasidirect:
            mf.get_jk = frag.impham_get_jk 
        else:
            mf._eri = ao2mo.restore(8, frag.impham_TEI, frag.norbs_imp)
        mf.scf( guess_1RDM )
        if ( mf.converged == False ):
            mf = mf.newton ()
            mf.kernel ()

    oneRDMimp_imp = mf.make_rdm1()    
    print ("Maximum distance between oneRDMimp_imp and guess_1RDM: {}".format (np.amax (np.abs (oneRDMimp_imp - guess_1RDM))))

    frag.oneRDM_loc = symmetrize_tensor (frag.oneRDMfroz_loc + represent_operator_in_basis (oneRDMimp_imp, frag.imp2loc))
    frag.twoCDM_imp = None
    frag.E_imp      = frag.impham_CONST + mf.e_tot + np.einsum ('ab,ab->', oneRDMimp_imp, chempot_imp)
    frag.loc2mo     = np.dot (frag.loc2imp, mf.mo_coeff)

    print ("Time for impurity RHF: {} seconds".format (time.time () - t_start))

    return None

