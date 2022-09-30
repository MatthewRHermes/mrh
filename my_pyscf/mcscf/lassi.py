import numpy as np
import time
from scipy import linalg
from mrh.my_pyscf.mcscf import lassi_op_o0 as op_o0
from mrh.my_pyscf.mcscf import lassi_op_o1 as op_o1
from pyscf import lib, symm
from pyscf.lib.numpy_helper import tag_array
from pyscf.fci.direct_spin1 import _unpack_nelec
from itertools import combinations, product

LINDEP_THRESHOLD = 1.0e-5

op = (op_o0, op_o1)

def ham_2q (las, mo_coeff, veff_c=None, h2eff_sub=None):
    # Construct second-quantization Hamiltonian
    ncore, ncas, nocc = las.ncore, las.ncas, las.ncore + las.ncas
    mo_core = mo_coeff[:,:ncore]
    mo_cas = mo_coeff[:,ncore:nocc]
    hcore = las._scf.get_hcore ()
    if veff_c is None: 
        dm_core = 2 * mo_core @ mo_core.conj ().T
        veff_c = las.get_veff (dm1s=dm_core)
    if h2eff_sub is None:
        h2eff_sub = las.ao2mo (mo_coeff)
    e0 = las._scf.energy_nuc () + 2 * (((hcore + veff_c/2) @ mo_core) * mo_core).sum ()
    h1 = mo_cas.conj ().T @ (hcore + veff_c) @ mo_cas
    h2 = h2eff_sub[ncore:nocc].reshape (ncas*ncas, ncas * (ncas+1) // 2)
    h2 = lib.numpy_helper.unpack_tril (h2).reshape (ncas, ncas, ncas, ncas)
    return e0, h1, h2

def las_symm_tuple (las, soc):
    # This really should be much more modular
    # If soc == False, symmetry tuple: neleca, nelecb, irrep
    # If soc == True, symmetry tuple: (neleca+nelecb), irrep
    statesym = []
    s2_states = []
    for iroot in range (las.nroots):
        neleca = 0
        nelecb = 0
        wfnsym = 0
        s = 0
        m = []
        for fcibox, nelec in zip (las.fciboxes, las.nelecas_sub):
            solver = fcibox.fcisolvers[iroot]
            na, nb = _unpack_nelec (fcibox._get_nelec (solver, nelec))
            neleca += na
            nelecb += nb
            s_frag = (solver.smult - 1) // 2
            s += s_frag * (s_frag + 1)
            m.append ((na-nb)//2)
            fragsym = getattr (solver, 'wfnsym', 0) or 0 # in case getattr returns "None"
            if isinstance (fragsym, str):
                fragsym = symm.irrep_name2id (solver.mol.groupname, fragsym)
            assert isinstance (fragsym, (int, np.integer)), '{} {}'.format (type (fragsym), fragsym)
            wfnsym ^= fragsym
        s += sum ([2*m1*m2 for m1, m2 in combinations (m, 2)])
        s2_states.append (s)
        if soc == False:
            statesym.append ((neleca, nelecb, wfnsym))
        else: 
            statesym.append ((neleca+nelecb, wfnsym))
    lib.logger.info (las, 'Symmetry analysis of LAS states:')
    lib.logger.info (las, ' {:2s}  {:>16s}  {:6s}  {:6s}  {:6s}  {:6s}'.format ('ix', 'Energy', 'Neleca', 'Nelecb', '<S**2>', 'Wfnsym'))
    for ix, (e, sy, s2) in enumerate (zip (las.e_states, statesym, s2_states)):
        if soc == False:
            neleca, nelecb, wfnsym = sy
        else: 
            nelec, wfnsym = sy
        wfnsym = symm.irrep_id2name (las.mol.groupname, wfnsym)
        lib.logger.info (las, ' {:2d}  {:16.10f}  {:6d}  {:6d}  {:6.3f}  {:>6s}'.format (ix, e, neleca, nelecb, s2, wfnsym))

    return statesym, np.asarray (s2_states)

def lassi (las, mo_coeff=None, ci=None, veff_c=None, h2eff_sub=None, orbsym=None, soc=False, opt=1):
    ''' Diagonalize the state-interaction matrix of LASSCF '''
    if mo_coeff is None: mo_coeff = las.mo_coeff
    if ci is None: ci = las.ci
    if orbsym is None: 
        orbsym = getattr (las.mo_coeff, 'orbsym', None)
        if orbsym is None and callable (getattr (las, 'label_symmetry_', None)):
            orbsym = las.label_symmetry_(las.mo_coeff).orbsym
        if orbsym is not None:
            orbsym = orbsym[las.ncore:las.ncore+las.ncas]
    o0_memcheck = op_o0.memcheck (las, ci)
    if opt == 0 and o0_memcheck == False:
        raise RuntimeError ('Insufficient memory to use o0 LASSI algorithm')

    # Construct second-quantization Hamiltonian
    e0, h1, h2 = ham_2q (las, mo_coeff, veff_c=None, h2eff_sub=None)

    # Symmetry tuple: neleca, nelecb, irrep
    statesym, s2_states = las_symm_tuple (las, soc)

    # Loop over symmetry blocks
    e_roots = np.zeros (las.nroots, dtype=np.float64)

    if soc == False:
        s2_roots = np.zeros (las.nroots, dtype=np.float64)
        si = np.zeros ((las.nroots, las.nroots), dtype=np.float64)
        s2_mat = np.zeros ((las.nroots, las.nroots), dtype=np.float64)
    else:
        s2_roots = np.zeros (las.nroots, dtype=complex)
        si = np.zeros ((las.nroots, las.nroots), dtype=complex)
        s2_mat = np.zeros ((las.nroots, las.nroots), dtype=complex)        
    
    for rootsym in set (statesym):
        idx = np.all (np.array (statesym) == rootsym, axis=1)
        lib.logger.debug (las, 'Diagonalizing LAS state symmetry block (neleca, nelecb, irrep) = {}'.format (rootsym))
        if np.count_nonzero (idx) == 1:
            lib.logger.debug (las, 'Only one state in this symmetry block')
            e_roots[idx] = las.e_states[idx] - e0
            si[np.ix_(idx,idx)] = 1.0
            s2_roots[idx] = s2_states[idx]
            continue
        wfnsym = rootsym[-1]
        ci_blk = [[c for c, ix in zip (cr, idx) if ix] for cr in ci]
        t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
        if (las.verbose > lib.logger.INFO) and (o0_memcheck):
            ham_ref, s2_ref, ovlp_ref = op_o0.ham (las, h1, h2, ci_blk, idx, soc=soc, orbsym=orbsym, wfnsym=wfnsym)
            t0 = lib.logger.timer (las, 'LASSI diagonalizer rootsym {} CI algorithm'.format (rootsym), *t0)
            ham_blk, s2_blk, ovlp_blk = op_o1.ham (las, h1, h2, ci_blk, idx, orbsym=orbsym, wfnsym=wfnsym)
            t0 = lib.logger.timer (las, 'LASSI diagonalizer rootsym {} TDM algorithm'.format (rootsym), *t0)
            lib.logger.debug (las, 'LASSI diagonalizer rootsym {}: ham o0-o1 algorithm disagreement = {}'.format (rootsym, linalg.norm (ham_blk - ham_ref))) 
            lib.logger.debug (las, 'LASSI diagonalizer rootsym {}: S2 o0-o1 algorithm disagreement = {}'.format (rootsym, linalg.norm (s2_blk - s2_ref))) 
            lib.logger.debug (las, 'LASSI diagonalizer rootsym {}: ovlp o0-o1 algorithm disagreement = {}'.format (rootsym, linalg.norm (ovlp_blk - ovlp_ref))) 
            errvec = np.concatenate ([(ham_blk-ham_ref).ravel (), (s2_blk-s2_ref).ravel (), (ovlp_blk-ovlp_ref).ravel ()])
            if np.amax (np.abs (errvec)) > 1e-8 and soc == False: # tmp until SOC is implemented for op_o1
                raise RuntimeError (("Congratulations, you have found a bug in either lassi_op_o0 (I really hope not)"
                    " or lassi_op_o1 (much more likely)!\nPlease inspect the last few printed lines of logger output"
                    " for more information.\nError in lassi, max abs: {}; norm: {}").format (np.amax (np.abs (errvec)),
                    linalg.norm (errvec)))
            if opt == 0:
                ham_blk = ham_ref
                s2_blk = s2_ref
                ovlp_blk = ovlp_ref
        else:
            if (las.verbose > lib.logger.INFO): lib.logger.debug (las, 'Insufficient memory to test against o0 LASSI algorithm')
            ham_blk, s2_blk, ovlp_blk = op[opt].ham (las, h1, h2, ci_blk, idx, soc=soc, orbsym=orbsym, wfnsym=wfnsym)
            t0 = lib.logger.timer (las, 'LASSI H build rootsym {}'.format (rootsym), *t0)
        lib.logger.debug2 (las, 'Block Hamiltonian - ecore:')
        lib.logger.debug2 (las, '{}'.format (ham_blk))
        lib.logger.debug2 (las, 'Block S**2:')
        lib.logger.debug2 (las, '{}'.format (s2_blk))
        lib.logger.debug2 (las, 'Block overlap matrix:')
        lib.logger.debug2 (las, '{}'.format (ovlp_blk))
        s2_mat[np.ix_(idx,idx)] = s2_blk
        # Error catch: diagonal Hamiltonian elements
        diag_test = np.diag (ham_blk)
        diag_ref = las.e_states[idx] - e0
        maxerr = np.max (np.abs (diag_test-diag_ref))
        if maxerr>1e-5 and soc == False: # tmp?
            lib.logger.debug (las, '{:>13s} {:>13s} {:>13s}'.format ('Diagonal', 'Reference', 'Error'))
            for ix, (test, ref) in enumerate (zip (diag_test, diag_ref)):
                lib.logger.debug (las, '{:13.6e} {:13.6e} {:13.6e}'.format (test, ref, test-ref))
            raise RuntimeError ('SI Hamiltonian diagonal element error = {}'.format (maxerr))
        # Error catch: linear dependencies in basis
        try:
            e, c = linalg.eigh (ham_blk, b=ovlp_blk)
        except linalg.LinAlgError as e:
            ovlp_det = linalg.det (ovlp_blk)
            lc = 'checking if LASSI basis has lindeps: |ovlp| = {:.6e}'.format (ovlp_det)
            lib.logger.info (las, 'Caught error %s, %s', str (e), lc)
            if ovlp_det < LINDEP_THRESHOLD:
                err_str = ('LASSI basis appears to have linear dependencies; '
                           'double-check your state list.\n'
                           '|ovlp| = {:.6e}').format (ovlp_det)
                raise RuntimeError (err_str) from e
            else: raise (e) from None
        s2_blk = c.conj ().T @ s2_blk @ c
        lib.logger.debug2 (las, 'Block S**2 in adiabat basis:')
        lib.logger.debug2 (las, '{}'.format (s2_blk))
        e_roots[idx] = e
        s2_roots[idx] = np.diag (s2_blk)
        si[np.ix_(idx,idx)] = c
    idx = np.argsort (e_roots)
    rootsym = np.array (statesym)[idx]
    e_roots = e_roots[idx] + e0
    s2_roots = s2_roots[idx]
    if soc == False:
        nelec_roots = [statesym[ix][0:2] for ix in idx]
        wfnsym_roots = [statesym[ix][2] for ix in idx]
    else:
        nelec_roots = [statesym[ix][0] for ix in idx]
        wfnsym_roots = [statesym[ix][1] for ix in idx]    
    si = si[:,idx]
    si = tag_array (si, s2=s2_roots, s2_mat=s2_mat, nelec=nelec_roots, wfnsym=wfnsym_roots)
    lib.logger.info (las, 'LASSI eigenvalues:')
    lib.logger.info (las, ' {:2s}  {:>16s}  {:6s}  {:6s}  {:6s}  {:6s}'.format ('ix', 'Energy', 'Neleca', 'Nelecb', '<S**2>', 'Wfnsym'))
    for ix, (er, s2r, rsym) in enumerate (zip (e_roots, s2_roots, rootsym)):
        if soc == False:  
            neleca, nelecb, wfnsym = rsym
            wfnsym = symm.irrep_id2name (las.mol.groupname, wfnsym)
            lib.logger.info (las, ' {:2d}  {:16.10f}  {:6d}  {:6d}  {:6.3f}  {:>6s}'.format (ix, er, neleca, nelecb, s2r, wfnsym))
        else:
            nelec, wfnsym = rsym
            wfnsym = symm.irrep_id2name (las.mol.groupname, wfnsym)
            lib.logger.info (las, ' {:2d}  {:16.10f}  {:6d}  {:6.3f}  {:>6s}'.format (ix, er, nelec, s2r, wfnsym))
    return e_roots, si

def make_stdm12s (las, ci=None, orbsym=None, soc=False, opt=1):
    ''' Evaluate <I|p'q|J> and <I|p'r'sq|J> where |I>, |J> are LAS states.

        Args:
            las: LASCI object

        Kwargs:
            ci: list of list of ci vectors
            orbsym: None or list of orbital symmetries spanning the whole orbital space
            opt: Optimization level, i.e.,  take outer product of
                0: CI vectors
                1: TDMs

        Returns:
            stdm1s: ndarray of shape (nroots,2,ncas,ncas,nroots)
            stdm2s: ndarray of shape (nroots,2,ncas,ncas,2,ncas,ncas,nroots)
    '''
    if ci is None: ci = las.ci
    if orbsym is None: 
        orbsym = getattr (las.mo_coeff, 'orbsym', None)
        if orbsym is None and callable (getattr (las, 'label_symmetry_', None)):
            orbsym = las.label_symmetry_(las.mo_coeff).orbsym
        if orbsym is not None:
            orbsym = orbsym[las.ncore:las.ncore+las.ncas]
    o0_memcheck = op_o0.memcheck (las, ci)
    if opt == 0 and o0_memcheck == False:
        raise RuntimeError ('Insufficient memory to use o0 LASSI algorithm')

    norb = las.ncas
    statesym = las_symm_tuple (las, soc)[0]
    stdm1s = np.zeros ((las.nroots, las.nroots, 2, norb, norb),
        dtype=ci[0][0].dtype).transpose (0,2,3,4,1)
    stdm2s = np.zeros ((las.nroots, las.nroots, 2, norb, norb, 2, norb, norb),
        dtype=ci[0][0].dtype).transpose (0,2,3,4,5,6,7,1)

    for rootsym in set (statesym):
        idx = np.all (np.array (statesym) == rootsym, axis=1)
        wfnsym = rootsym[-1]
        ci_blk = [[c for c, ix in zip (cr, idx) if ix] for cr in ci]
        t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
        if (las.verbose > lib.logger.INFO) and (o0_memcheck):
            d1s, d2s = op_o0.make_stdm12s (las, ci_blk, idx, soc=soc, orbsym=orbsym, wfnsym=wfnsym)
            t0 = lib.logger.timer (las, 'LASSI make_stdm12s rootsym {} CI algorithm'.format (rootsym), *t0)
            d1s_test, d2s_test = op_o1.make_stdm12s (las, ci_blk, idx)
            t0 = lib.logger.timer (las, 'LASSI make_stdm12s rootsym {} TDM algorithm'.format (rootsym), *t0)
            lib.logger.debug (las, 'LASSI make_stdm12s rootsym {}: D1 o0-o1 algorithm disagreement = {}'.format (rootsym, linalg.norm (d1s_test - d1s))) 
            lib.logger.debug (las, 'LASSI make_stdm12s rootsym {}: D2 o0-o1 algorithm disagreement = {}'.format (rootsym, linalg.norm (d2s_test - d2s))) 
            errvec = np.concatenate ([(d1s-d1s_test).ravel (), (d2s-d2s_test).ravel ()])
            if np.amax (np.abs (errvec)) > 1e-8 and soc == False: # tmp until SOC implemented for op_o1
                raise RuntimeError (("Congratulations, you have found a bug in either lassi_op_o0 (I really hope not)"
                    " or lassi_op_o1 (much more likely)!\nPlease inspect the last few printed lines of logger output"
                    " for more information.\nError in make_stdm12s, max abs: {}; norm: {}").format (np.amax (np.abs (errvec)),
                    linalg.norm (errvec)))
            if opt == 0:
                d1s = d1s_test
                d2s = d2s_test
        else:
            if (las.verbose > lib.logger.INFO): lib.logger.debug (las, 'Insufficient memory to test against o0 LASSI algorithm')
            if opt == 0: # tmp until SOC implemented for op_o1
                d1s, d2s = op_o0.make_stdm12s (las, ci_blk, idx, soc=soc, orbsym=orbsym, wfnsym=wfnsym)
            else:
                d1s, d2s = op_o1.make_stdm12s (las, ci_blk, idx, orbsym=orbsym, wfnsym=wfnsym)
            #d1s, d2s = op[opt].make_stdm12s (las, ci_blk, idx, orbsym=orbsym, wfnsym=wfnsym)
            t0 = lib.logger.timer (las, 'LASSI make_stdm12s rootsym {}'.format (rootsym), *t0)
        idx_int = np.where (idx)[0]
        for (i,a), (j,b) in product (enumerate (idx_int), repeat=2):
            stdm1s[a,...,b] = d1s[i,...,j]
            stdm2s[a,...,b] = d2s[i,...,j]
    return stdm1s, stdm2s

def roots_make_rdm12s (las, ci, si, soc=False, orbsym=None, opt=1):
    if orbsym is None: 
        orbsym = getattr (las.mo_coeff, 'orbsym', None)
        if orbsym is None and callable (getattr (las, 'label_symmetry_', None)):
            orbsym = las.label_symmetry_(las.mo_coeff).orbsym
        if orbsym is not None:
            orbsym = orbsym[las.ncore:las.ncore+las.ncas]
    o0_memcheck = op_o0.memcheck (las, ci)
    if opt == 0 and o0_memcheck == False:
        raise RuntimeError ('Insufficient memory to use o0 LASSI algorithm')

    # Symmetry tuple: neleca, nelecb, irrep
    norb = las.ncas
    statesym = las_symm_tuple (las, soc)[0]
    rdm1s = np.zeros ((las.nroots, 2, norb, norb),
        dtype=ci[0][0].dtype)
    rdm2s = np.zeros ((las.nroots, 2, norb, norb, 2, norb, norb),
        dtype=ci[0][0].dtype)
    rootsym = [(ne[0], ne[1], wfnsym) for ne, wfnsym in zip (si.nelec, si.wfnsym)]

    for sym in set (statesym):
        idx_ci = np.all (np.array (statesym) == sym, axis=1)
        idx_si = np.all (np.array (rootsym)  == sym, axis=1)
        wfnsym = sym[-1]
        ci_blk = [[c for c, ix in zip (cr, idx_ci) if ix] for cr in ci]
        si_blk = si[np.ix_(idx_ci,idx_si)]
        t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
        if (las.verbose > lib.logger.INFO) and (o0_memcheck):
            d1s, d2s = op_o0.roots_make_rdm12s (las, ci_blk, idx_ci, si_blk, soc=soc, orbsym=orbsym, wfnsym=wfnsym)
            t0 = lib.logger.timer (las, 'LASSI make_rdm12s rootsym {} CI algorithm'.format (sym), *t0)
            d1s_test, d2s_test = op_o1.roots_make_rdm12s (las, ci_blk, idx_ci, si_blk)
            t0 = lib.logger.timer (las, 'LASSI make_rdm12s rootsym {} TDM algorithm'.format (sym), *t0)
            lib.logger.debug (las, 'LASSI make_rdm12s rootsym {}: D1 o0-o1 algorithm disagreement = {}'.format (sym, linalg.norm (d1s_test - d1s))) 
            lib.logger.debug (las, 'LASSI make_rdm12s rootsym {}: D2 o0-o1 algorithm disagreement = {}'.format (sym, linalg.norm (d2s_test - d2s))) 
            errvec = np.concatenate ([(d1s-d1s_test).ravel (), (d2s-d2s_test).ravel ()])
            if np.amax (np.abs (errvec)) > 1e-8 and soc == False: # tmp until SOC is implemented for op_o1
                raise RuntimeError (("Congratulations, you have found a bug in either lassi_op_o0 (I really hope not)"
                    " or lassi_op_o1 (much more likely)!\nPlease inspect the last few printed lines of logger output"
                    " for more information.\nError in make_stdm12s, max abs: {}; norm: {}").format (np.amax (np.abs (errvec)),
                    linalg.norm (errvec)))
            if opt == 0:
                d1s = d1s_test
                d2s = d2s_test
        else:
            if (las.verbose > lib.logger.INFO): lib.logger.debug (las, 'Insufficient memory to test against o0 LASSI algorithm')
            d1s, d2s = op[opt].roots_make_rdm12s (las, ci_blk, idx_ci, si_blk, orbsym=orbsym, wfnsym=wfnsym)
            t0 = lib.logger.timer (las, 'LASSI make_rdm12s rootsym {}'.format (sym), *t0)
        idx_int = np.where (idx_si)[0]
        for (i,a) in enumerate (idx_int):
            rdm1s[a] = d1s[i]
            rdm2s[a] = d2s[i]
    return rdm1s, rdm2s






