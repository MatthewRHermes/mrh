import numpy as np
import time
from scipy import linalg
from mrh.my_pyscf.lassi import op_o0
from mrh.my_pyscf.lassi import op_o1
from mrh.my_pyscf.lassi import basis
from mrh.my_pyscf.lassi.citools import get_lroots
from pyscf import lib 
from pyscf.lib import param
from pyscf.scf.addons import canonical_orth_
from pyscf import __config__

LINDEP_THRESH = getattr (__config__, 'lassi_lindep_thresh', 1.0e-5)
MAX_CYCLE_SI = getattr (__config__, 'lassi_max_cycle_si', 100)
MAX_SPACE_SI = getattr (__config__, 'lassi_max_space_si', 12)
TOL_SI = getattr (__config__, 'lassi_tol_si', 1e-8)
LEVEL_SHIFT_SI = getattr (__config__, 'lassi_level_shift_si', 1.0e-8)
NROOTS_SI = getattr (__config__, 'lassi_nroots_si', 1)
DAVIDSON_SCREEN_THRESH_SI = getattr (__config__, 'lassi_hsi_screen_thresh', 1e-12)
PSPACE_SIZE_SI = getattr (__config__, 'lassi_hsi_pspace_size', 400)
PRIVREF_SI = getattr (__config__, 'lassi_privref_si', True)

op = (op_o0, op_o1)

def _eig_block (las, e0, h1, h2, ci_blk, nelec_blk, smult_blk, disc_blk, soc, opt,
                max_memory=param.MAX_MEMORY, davidson_only=False):
    nstates = np.prod (get_lroots (ci_blk), axis=0).sum ()
    req_memory = 24*nstates*nstates/1e6
    current_memory = lib.current_memory ()[0]
    if current_memory+req_memory > max_memory:
        if opt==0:
            raise MemoryError ("Need %f MB of %f MB av (N.B.: o0 Davidson is fake; use opt=1)",
                               req_memory, max_memory-current_memory)
        lib.logger.info (las, "Need %f MB of %f MB av for incore LASSI diag; Davidson alg forced",
                         req_memory, max_memory-current_memory)
    if davidson_only or current_memory+req_memory > max_memory:
        return _eig_block_Davidson (las, e0, h1, h2, ci_blk, nelec_blk, smult_blk, disc_blk, soc,
                                    opt)
    return _eig_block_incore (las, e0, h1, h2, ci_blk, nelec_blk, smult_blk, soc, opt)

def _eig_block_Davidson (las, e0, h1, h2, ci_blk, nelec_blk, smult_blk, disc_blk, soc, opt):
    # si0
    # nroots_si
    # level_shift
    verbose = las.verbose
    davidson_log = log = lib.logger.new_logger (las, verbose)
    # We want this Davidson diagonalizer to be louder than usual
    if verbose >= lib.logger.NOTE:
        davidson_log = lib.logger.new_logger (las, verbose+1)
    si0 = getattr (las, 'si', None)
    level_shift = getattr (las, 'level_shift_si', LEVEL_SHIFT_SI)
    nroots_si = getattr (las, 'nroots_si', NROOTS_SI)
    max_cycle_si = getattr (las, 'max_cycle_si', MAX_CYCLE_SI)
    max_space_si = getattr (las, 'max_space_si', MAX_SPACE_SI)
    tol_si = getattr (las, 'tol_si', TOL_SI)
    get_init_guess = getattr (las, 'get_init_guess_si', get_init_guess_si)
    privilege_ref = getattr (las, 'privref_si', PRIVREF_SI)
    screen_thresh = getattr (las, 'davidson_screen_thresh_si', DAVIDSON_SCREEN_THRESH_SI)
    pspace_size = getattr (las, 'pspace_size_si', PSPACE_SIZE_SI)
    smult_si = getattr (las, 'smult_si', None)
    h_op_raw, s2_op, ovlp_op, hdiag_raw, _get_ovlp = op[opt].gen_contract_op_si_hdiag (
        las, h1, h2, ci_blk, nelec_blk, smult_fr=smult_blk, soc=soc, disc_fr=disc_blk,
        screen_thresh=screen_thresh
    )
    if verbose >= lib.logger.DEBUG:
        # The sort is slow
        log.debug ("fingerprint of hdiag raw: %15.10e", lib.fp (np.sort (hdiag_raw)))
    t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
    raw2orth = basis.get_orth_basis (ci_blk, las.ncas_sub, nelec_blk, _get_ovlp=_get_ovlp,
                                     smult_fr=smult_blk, smult_si=smult_si, disc_fr=disc_blk)
    raw2orth.log_debug1_hdiag_raw (log, hdiag_raw)
    orth2raw = raw2orth.H
    mem_orth = raw2orth.get_nbytes () / 1e6
    t0 = log.timer ('LASSI get orthogonal basis ({:.2f} MB)'.format (mem_orth), *t0)
    hdiag_orth = op[opt].get_hdiag_orth (hdiag_raw, h_op_raw, raw2orth)
    if verbose >= lib.logger.DEBUG:
        # The sort is slow
        log.debug ("fingerprint of hdiag orth: %15.10e", lib.fp (np.sort (hdiag_orth)))
    t0 = log.timer ('LASSI get hdiag in orthogonal basis', *t0)
    hdiag_penalty = np.zeros_like (hdiag_orth)
    if privilege_ref:
        # Force the reference state to appear in the first (few?) guess vectors
        i = raw2orth.get_ref_man_size ()
        if (i>0) and (i < len (hdiag_orth)):
            below = np.amin (hdiag_orth[i:])
            above = np.amax (hdiag_orth[:i])
            if above > below:
                penvalue = above - below + 0.001
                log.debug ("Hdiag penalty value: %17.10e", penvalue)
                hdiag_penalty[i:] = penvalue
    if pspace_size:
        pw, pv, addr = pspace (hdiag_orth, h_op_raw, raw2orth, opt, pspace_size, log=log,
                               penalty=hdiag_penalty)
        t0 = log.timer ('LASSI make pspace Hamiltonian', *t0)
        if pspace_size >= hdiag_orth.size:
            pv = pv[:,:nroots_si]
            pw = pw[:nroots_si]
            si1 = orth2raw (pv)
            s2 = lib.einsum ('ij,ij->j', si1.conj (), s2_op (si1))
            return True, pw, si1, s2
        precond_op = make_pspace_precond (hdiag_orth, pw, pv, addr, level_shift=level_shift)
    else:
        precond_op = lib.make_diag_precond (hdiag_orth, level_shift=level_shift)
    if si0 is not None:
        x0 = raw2orth (ovlp_op (si0))
    else:
        x0 = None
    x0 = get_init_guess (hdiag_orth, nroots_si, x0, log=log, penalty=hdiag_penalty)
    def h_op (x):
        return raw2orth (h_op_raw (orth2raw (x)))
    log.info ("LASSI E(const) = %15.10f", e0)
    conv, e, x1 = lib.davidson1 (lambda xs: [h_op (x) for x in xs],
                                 x0, precond_op, nroots=nroots_si,
                                 verbose=davidson_log, max_cycle=max_cycle_si,
                                 max_space=max_space_si, tol=tol_si)
    conv = all (conv)
    if not conv: log.warn ('LASSI Davidson diagonalization not converged')
    si1 = np.stack ([orth2raw (x) for x in x1], axis=-1)
    s2 = np.array ([np.dot (x.conj (), s2_op (x)) for x in si1.T])
    return conv, e, si1, s2

def pspace (hdiag_orth, h_op_raw, raw2orth, opt, pspace_size, log=None, penalty=None):
    heff = hdiag_orth.copy ()
    if penalty is not None:
        heff += penalty
    if hdiag_orth.size <= pspace_size:
        addr = np.arange (hdiag_orth.size)
    else:
        try: # this is just a fast PARTIAL sort
            addr = np.argpartition(heff, pspace_size-1)[:pspace_size].copy()
        except AttributeError:
            addr = np.argsort(heff)[:pspace_size].copy()
    h0 = op[opt].pspace_ham (h_op_raw, raw2orth, addr)
    pw, pv = linalg.eigh (h0)
    if log is not None:
        raw2orth.log_debug_hdiag_orth (log, hdiag_orth, idx=addr)
    e_pspace = h0.diagonal ()
    e_hdiag = hdiag_orth[addr]
    idx_err = np.abs (e_hdiag-e_pspace) > 1e-5
    if (log is not None) and (log.verbose > lib.logger.DEBUG) and (np.count_nonzero (idx_err)):
        # Some notes:
        # 1. For my small helium tetrahedron, pspace also fails for the lindep-affected states
        # 2. The 2-fragment soc failure of this seems to oscillate between just a few numbers,
        #    which is a pretty big hint.
        log.error ("LASSI hdiag and pspace Hamiltonian disagree!")
        log.error ("The incoming table may take a very long time to print out.")
        log.error ("Do not expect this calculation to complete.")
        log.error ("{:>4s} {:>17s} {:>17s} {:>17s}".format ('ix', 'pspace', 'hdiag', 'operator'))
        fmt_str = '{:4d} {:17.10e} {:17.10e} {:17.10e}'
        for i in np.where (idx_err)[0]:
            x = np.zeros (raw2orth.shape[0], dtype=raw2orth.dtype)
            x[addr[i]] = 1.0
            x = raw2orth.H (x)
            e_ref = np.dot (x.conj (), h_op_raw (x))
            log.error (fmt_str.format (addr[i], e_pspace[i], e_hdiag[i], e_ref))
        raise RuntimeError ("LASSI hdiag and pspace Hamiltonian disagree!")
    return pw, pv, addr

def make_pspace_precond(hdiag, pspaceig, pspaceci, addr, level_shift=0):
    # precondition with pspace Hamiltonian, CPL, 169, 463
    # copied and modified from PySCF d57cb6d6c722bcc28c5db8573a75bb6bc67a8583
    def get_hinv (e0):
        h0e0inv = np.dot(pspaceci/(pspaceig-(e0-level_shift)), pspaceci.T)
        hdiaginv = 1/(hdiag - (e0-level_shift))
        hdiaginv[abs(hdiaginv)>1e8] = 1e8
        def hinv (x0):
            x1 = hdiaginv * x0
            x1[addr] = np.dot (h0e0inv, x0[addr])
            return x1
        return hinv
    def precond(r, e0, x0, *args):
        hinv = get_hinv (e0)
        h0x0 = hinv (x0)
        h0r = hinv (r)
        e1 = np.dot(x0, h0r) / np.dot(x0, h0x0)
        x1 = hinv (r - e1*x0)
        return x1
    return precond

def get_init_guess_si (hdiag, nroots, si1, log=None, penalty=None):
    nprod = hdiag.size
    heff = hdiag.copy ()
    if penalty is not None:
        heff += penalty
    si0 = []
    if nprod <= nroots:
        addrs = np.arange(nprod)
    else:
        addrs = np.argpartition(heff, nroots-1)[:nroots]
    for addr in addrs:
        x = np.zeros((nprod))
        x[addr] = 1
        si0.append(x)
    # Add noise
    si0[0][0 ] += 1e-5
    si0[0][-1] -= 1e-5
    j = 0
    if si1 is not None:
        si1 = si1.reshape (nprod,-1)
        j = si1.shape[1]
        for i in range (min (j, nroots)):
            si0[i] = si1[:,i]
    if (j < nroots) and (log is not None):
        log.info ('Energy of guess SI vectors: {}'.format (
            hdiag[addrs[j:]]
        ))
    return si0

def _eig_block_incore (las, e0, h1, h2, ci_blk, nelec_blk, smult_blk, soc, opt):
    # TODO: simplify
    t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
    o0_memcheck = op_o0.memcheck (las, ci_blk, soc=soc)
    if (las.verbose > lib.logger.INFO) and (o0_memcheck):
        ham_ref, s2_ref, ovlp_ref = op_o0.ham (las, h1, h2, ci_blk, nelec_blk, soc=soc)[:3]
        t0 = lib.logger.timer (las, 'LASSI diagonalizer CI algorithm', *t0)

        h1_sf = h1
        if soc:
            h1_sf = (h1[0:las.ncas,0:las.ncas]
                     - h1[las.ncas:2*las.ncas,las.ncas:2*las.ncas]).real/2
        ham_blk, s2_blk, ovlp_blk, _get_ovlp = op[opt].ham (
            las, h1_sf, h2, ci_blk, nelec_blk)
        t0 = lib.logger.timer (las, 'LASSI diagonalizer TDM algorithm', *t0)
        lib.logger.debug (las,
            'LASSI diagonalizer ham o0-o1 algorithm disagreement = {}'.format (
                linalg.norm (ham_blk - ham_ref))) 
        lib.logger.debug (las,
            'LASSI diagonalizer S2 o0-o1 algorithm disagreement = {}'.format (
                linalg.norm (s2_blk - s2_ref))) 
        lib.logger.debug (las,
            'LASSI diagonalizer ovlp o0-o1 algorithm disagreement = {}'.format (
                linalg.norm (ovlp_blk - ovlp_ref))) 
        errvec = np.concatenate ([(ham_blk-ham_ref).ravel (), (s2_blk-s2_ref).ravel (),
                                  (ovlp_blk-ovlp_ref).ravel ()])
        if np.amax (np.abs (errvec)) > 1e-8 and soc == False: # tmp until SOC in op_o1
            raise LASSIOop01DisagreementError ("Hamiltonian + S2 + Ovlp", errvec)
        if opt == 0:
            ham_blk = ham_ref
            s2_blk = s2_ref
            ovlp_blk = ovlp_ref
    else:
        if (las.verbose > lib.logger.INFO): lib.logger.debug (
            las, 'Insufficient memory to test against o0 LASSI algorithm')
        ham_blk, s2_blk, ovlp_blk, _get_ovlp = op[opt].ham (
            las, h1, h2, ci_blk, nelec_blk, smult_fr=smult_blk, soc=soc)
        t0 = lib.logger.timer (las, 'LASSI H build', *t0)
    log_debug = lib.logger.debug2 if las.nroots>10 else lib.logger.debug
    if np.iscomplexobj (ham_blk):
        log_debug (las, 'Block Hamiltonian - ecore (real):')
        log_debug (las, '{}'.format (ham_blk.real.round (8)))
        log_debug (las, 'Block Hamiltonian - ecore (imag):')
        log_debug (las, '{}'.format (ham_blk.imag.round (8)))
    else:
        log_debug (las, 'Block Hamiltonian - ecore:')
        log_debug (las, '{}'.format (ham_blk.round (8)))
    log_debug (las, 'Block S**2:')
    log_debug (las, '{}'.format (s2_blk.round (8)))
    log_debug (las, 'Block overlap matrix:')
    log_debug (las, '{}'.format (ovlp_blk.round (8)))
    # Error catch: diagonal Hamiltonian elements
    # This diagnostic is simply not valid for local excitations;
    # the energies aren't supposed to be additive
    lroots = get_lroots (ci_blk)
    e_states_meaningful = not getattr (las, 'e_states_meaningless', False)
    e_states_meaningful &= np.all (lroots==1)
    e_states_meaningful &= not (soc) # TODO: fix?
    if e_states_meaningful:
        diag_test = np.diag (ham_blk)
        diag_ref = las.e_states - e0
        maxerr = np.max (np.abs (diag_test-diag_ref))
        if maxerr>1e-5:
            lib.logger.debug (las, '{:>13s} {:>13s} {:>13s}'.format ('Diagonal', 'Reference',
                                                                     'Error'))
            for ix, (test, ref) in enumerate (zip (diag_test, diag_ref)):
                lib.logger.debug (las, '{:13.6e} {:13.6e} {:13.6e}'.format (test, ref, test-ref))
            lib.logger.warn (las, 'LAS states in basis may not be converged (%s = %e)',
                             'max(|Hdiag-e_states|)', maxerr)
    # Error catch: linear dependencies in basis
    raw2orth = basis.get_orth_basis (ci_blk, las.ncas_sub, nelec_blk, _get_ovlp=_get_ovlp,
                                     smult_fr=smult_blk)
    xhx = raw2orth (ham_blk.T).T
    lib.logger.info (las, '%d/%d linearly independent model states',
                     xhx.shape[1], xhx.shape[0])
    xhx = raw2orth (xhx.conj ()).conj ()
    try:
        e, c = linalg.eigh (xhx)
    except linalg.LinAlgError as err:
        ovlp_det = linalg.det (ovlp_blk)
        lc = 'checking if LASSI basis has lindeps: |ovlp| = {:.6e}'.format (ovlp_det)
        lib.logger.info (las, 'Caught error %s, %s', str (err), lc)
        if ovlp_det < LINDEP_THRESH:
            x_ref = canonical_orth_(ovlp_blk, thr=LINDEP_THRESH)
            x_test = raw2orth (np.eye (ham_blk.shape[0]))
            x_ovlp = x_test.conj () @ x_ref
            x_err = x_ovlp @ x_ovlp.conj ().T - np.eye (x_ovlp.shape[0])
            err = np.trace (x_err)
            raise RuntimeError ("LASSI lindep prescreening failure; orth err = {}".format (err))
        else: raise (err) from None
    c = raw2orth.H (c)
    s2_blk = ((s2_blk @ c) * c.conj ()).sum (0)
    return True, e, c, s2_blk

