import numpy as np
from scipy import linalg
from mrh.my_pyscf.mcscf.lassi_op_slow import slow_ham
from pyscf import lib, symm
from pyscf.lib.numpy_helper import tag_array
from pyscf.fci.direct_spin1 import _unpack_nelec
from itertools import combinations

def lassi (las, mo_coeff=None, ci=None, veff_c=None, h2eff_sub=None, orbsym=None):
    ''' Diagonalize the state-interaction matrix of LASSCF '''
    if mo_coeff is None: mo_coeff = las.mo_coeff
    if ci is None: ci = las.ci
    if orbsym is None: orbsym = getattr (mo_coeff, 'orbsym', None)

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

    # Symmetry tuple: neleca, nelecb, irrep
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
            fragsym = getattr (solver, 'wfnsym', 0)
            if isinstance (fragsym, str):
                fragsym = symm.irrep_name2id (las.mol.groupname, fragsym)
            wfnsym ^= fragsym
        s += sum ([2*m1*m2 for m1, m2 in combinations (m, 2)])
        s2_states.append (s)
        statesym.append ((neleca, nelecb, wfnsym))
    lib.logger.info (las, 'Symmetry analysis of LAS states:')
    lib.logger.info (las, ' {:2s}  {:>16s}  {:6s}  {:6s}  {:6s}  {:6s}'.format ('ix', 'Energy', 'Neleca', 'Nelecb', '<S**2>', 'Wfnsym'))
    for ix, (e, sy, s2) in enumerate (zip (las.e_states, statesym, s2_states)):
        neleca, nelecb, wfnsym = sy
        wfnsym = symm.irrep_id2name (las.mol.groupname, wfnsym)
        lib.logger.info (las, ' {:2d}  {:16.10f}  {:6d}  {:6d}  {:6.3f}  {:>6s}'.format (ix, e, neleca, nelecb, s2, wfnsym))

    # Loop over symmetry blocks
    e_roots = np.zeros (las.nroots, dtype=np.float64)
    s2_roots = np.zeros (las.nroots, dtype=np.float64)
    si = np.zeros ((las.nroots, las.nroots), dtype=np.float64)    
    for rootsym in set (statesym):
        idx = np.all (np.array (statesym) == rootsym, axis=1)
        lib.logger.debug (las, 'Diagonalizing LAS state symmetry block (neleca, nelecb, irrep) = {}'.format (rootsym))
        if np.count_nonzero (idx) == 1:
            lib.logger.debug (las, 'Only one state in this symmetry block')
            e_roots[idx] = las.e_states[idx]
            si[np.ix_(idx,idx)] = 1.0
            continue
        ci_blk = [[c for c, ix in zip (cr, idx) if ix] for cr in ci]
        nelec_blk = [[_unpack_nelec (fcibox._get_nelec (solver, nelecas)) for solver, ix in zip (fcibox.fcisolvers, idx)] for fcibox, nelecas in zip (las.fciboxes, las.nelecas_sub)]
        ham_blk, s2_blk, ovlp_blk = slow_ham (las.mol, h1, h2, ci_blk, las.ncas_sub, nelec_blk, orbsym=orbsym)
        lib.logger.debug (las, 'Block Hamiltonian - ecore:')
        lib.logger.debug (las, '{}'.format (ham_blk))
        lib.logger.debug (las, 'Block S**2:')
        lib.logger.debug (las, '{}'.format (s2_blk))
        lib.logger.debug (las, 'Block overlap matrix:')
        lib.logger.debug (las, '{}'.format (ovlp_blk))
        diag_test = np.diag (ham_blk)
        diag_ref = las.e_states[idx] - e0
        lib.logger.debug (las, '{:>13s} {:>13s} {:>13s}'.format ('Diagonal', 'Reference', 'Error'))
        for ix, (test, ref) in enumerate (zip (diag_test, diag_ref)):
            lib.logger.debug (las, '{:13.6e} {:13.6e} {:13.6e}'.format (test, ref, test-ref))
        assert (np.allclose (diag_test, diag_ref, atol=1e-5)), 'SI Hamiltonian diagonal element error. Inadequate convergence?'
        e, c = linalg.eigh (ham_blk, b=ovlp_blk)
        s2_blk = c.conj ().T @ s2_blk @ c
        lib.logger.debug (las, 'Block S**2 in adiabat basis:')
        lib.logger.debug (las, '{}'.format (s2_blk))
        e_roots[idx] = e
        s2_roots[idx] = np.diag (s2_blk)
        si[np.ix_(idx,idx)] = c
    idx = np.argsort (e_roots)
    rootsym = np.array (statesym)[idx]
    e_roots = e_roots[idx] + e0
    s2_roots = s2_roots[idx]
    nelec_roots = [statesym[ix][0:2] for ix in idx]
    wfnsym_roots = [statesym[ix][2] for ix in idx]
    si = si[:,idx]
    si = tag_array (si, s2=s2_roots, nelec=nelec_roots, wfnsym=wfnsym_roots)
    lib.logger.info (las, 'LASSI eigenvalues:')
    lib.logger.info (las, ' {:2s}  {:>16s}  {:6s}  {:6s}  {:6s}  {:6s}'.format ('ix', 'Energy', 'Neleca', 'Nelecb', '<S**2>', 'Wfnsym'))
    for ix, (er, s2r, rsym) in enumerate (zip (e_roots, s2_roots, rootsym)):
        neleca, nelecb, wfnsym = rsym
        wfnsym = symm.irrep_id2name (las.mol.groupname, wfnsym)
        lib.logger.info (las, ' {:2d}  {:16.10f}  {:6d}  {:6d}  {:6.3f}  {:>6s}'.format (ix, er, neleca, nelecb, s2r, wfnsym))
    return e_roots, si



