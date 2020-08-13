import sys
from pyscf import gto, dft, scf, mcscf, df
from pyscf.tools import molden
from mrh.my_dmet import localintegrals, dmet, fragments
from mrh.my_dmet.fragments import make_fragment_atom_list, make_fragment_orb_list
from pyscf.mcscf.addons import project_init_guess, spin_square, sort_mo_by_irrep
from pyscf.lib.parameters import BOHR
from functools import reduce
import numpy as np
import re
import me2n2_struct
import tracemalloc

def run (mf, CASlist=None, **kwargs):
    # I/O
    # --------------------------------------------------------------------------------------------------------------------
    mol = mf.mol
    my_kwargs = {'calcname':           'me2n2_lasscf',
                 'doLASSCF':           True,
                 'debug_energy':       False,
                 'debug_reloc':        False,
                 'nelec_int_thresh':   1e-5,
                 'num_mf_stab_checks': 0}
    bath_tol = 1e-8
    my_kwargs.update (kwargs)
    
    # Set up the localized AO basis
    # --------------------------------------------------------------------------------------------------------------------
    myInts = localintegrals.localintegrals(mf, range(mol.nao_nr ()), 'meta_lowdin')
    
    # Build fragments from atom list
    # --------------------------------------------------------------------------------------------------------------------
    N2 = make_fragment_atom_list (myInts, list (range(2)), 'CASSCF(4,4)', name="N2")
    N2.target_S = N2.target_MS = mol.spin // 2
    Me1 = make_fragment_atom_list (myInts, list (range(2,6)), 'RHF', name='Me1')
    Me2 = make_fragment_atom_list (myInts, list (range(6,10)), 'RHF', name='Me2')
    N2.bath_tol = Me1.bath_tol = Me2.bath_tol = bath_tol
    fraglist = [N2, Me1, Me2] 
    
    # Generate active orbital guess 
    # --------------------------------------------------------------------------------------------------------------------
    me2n2_dmet = dmet (myInts, fraglist, **my_kwargs)
    me2n2_dmet.generate_frag_cas_guess (mf.mo_coeff, caslst=CASlist, force_imp=True, confine_guess=False)
    
    # Calculation
    # --------------------------------------------------------------------------------------------------------------------
    return me2n2_dmet.doselfconsistent ()
    
    
