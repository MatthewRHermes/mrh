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

def run (r_nn, do_df=False, **kwargs):
    # I/O
    # --------------------------------------------------------------------------------------------------------------------
    basis = '6-31g'
    print ("Me2N2 at r_nn = {}".format (r_nn))
    CASlist = None #np.empty (0, dtype=int)
    ints = re.compile ("[0-9]+")
    my_kwargs = {'calcname':           'me2n2_lasscf_r{:2.0f}'.format (r_nn*10),
                 'doLASSCF':           True,
                 'debug_energy':       False,
                 'debug_reloc':        False,
                 'nelec_int_thresh':   1e-5,
                 'num_mf_stab_checks': 0}
    load_casscf_guess = False
    dr_guess = None
    bath_tol = 1e-8
    my_kwargs.update (kwargs)
    
    # Hartree--Fock calculation
    # --------------------------------------------------------------------------------------------------------------------
    mol = me2n2_struct.structure (r_nn, basis)
    mol.verbose = 0
    mf = scf.RHF (mol)
    if do_df:
        mf = mf.density_fit (auxbasis = df.aug_etb (mol))
    mf.kernel ()
    if not mf.converged:
        mf = mf.newton ()
        mf.kernel ()
    for i in range (my_kwargs['num_mf_stab_checks']):
        new_mo = mf.stability ()[0]
        dm0 = reduce (np.dot, (new_mo, np.diag (mf.mo_occ), new_mo.conjugate ().T))
        mf = scf.RHF (mol)
        mf.verbose = 4
        mf.kernel (dm0)
        if not mf.converged:
            mf = mf.newton ()
            mf.kernel ()
    
    # Set up the localized AO basis
    # --------------------------------------------------------------------------------------------------------------------
    myInts = localintegrals.localintegrals(mf, range(mol.nao_nr ()), 'meta_lowdin')
    myInts.molden( my_kwargs['calcname'] + '_locints.molden' )
    
    # Build fragments from atom list
    # --------------------------------------------------------------------------------------------------------------------
    N2 = make_fragment_atom_list (myInts, list (range(2)), 'CASSCF(4,4)', name="N2")
    Me1 = make_fragment_atom_list (myInts, list (range(2,6)), 'RHF', name='Me1')
    Me2 = make_fragment_atom_list (myInts, list (range(6,10)), 'RHF', name='Me2')
    N2.bath_tol = Me1.bath_tol = Me2.bath_tol = bath_tol
    fraglist = [N2, Me1, Me2] 
    
    # Load or generate active orbital guess 
    # --------------------------------------------------------------------------------------------------------------------
    me2n2_dmet = dmet (myInts, fraglist, **my_kwargs)
    if load_casscf_guess:
        npyfile = 'me2n2_casano.{:.1f}.npy'.format (r_nn)
        norbs_cmo = (mol.nelectron - 4) // 2
        norbs_amo = 4
        N2.load_amo_guess_from_casscf_npy (npyfile, norbs_cmo, norbs_amo)
    elif dr_guess is not None:
        chkname = 'me2n2_lasscf_r{:2.0f}'.format (dr_guess*10)
        me2n2_dmet.load_checkpoint (chkname + '.chk.npy')
    else:
        me2n2_dmet.generate_frag_cas_guess (mf.mo_coeff, caslst=CASlist)
    
    # Calculation
    # --------------------------------------------------------------------------------------------------------------------
    energy_result = me2n2_dmet.doselfconsistent ()
    print ("----Energy: {:.1f} {:.8f}".format (r_nn, energy_result))
    
    return energy_result
    
    
