import numpy as np
import time
from scipy import linalg
from mrh.my_pyscf.lassi import op_o0
from mrh.my_pyscf.lassi import op_o1
from mrh.my_pyscf.lassi import basis
from mrh.my_pyscf.lassi.citools import get_lroots
from pyscf import lib 
from pyscf.lib import param, logger
from pyscf.scf.addons import canonical_orth_
from pyscf import __config__

LINDEP_THRESH = getattr (__config__, 'lassi_lindep_thresh', 1.0e-5)
MAX_CYCLE = getattr (__config__, 'lassi_max_cycle_si', 100)
MAX_SPACE = getattr (__config__, 'lassi_max_space_si', 12)
CONV_TOL = getattr (__config__, 'lassi_conv_tol', getattr (__config__, 'lassi_tol_si', 1e-8))
LEVEL_SHIFT = getattr (__config__, 'lassi_level_shift_si', 1.0e-8)
NROOTS = getattr (__config__, 'lassi_nroots_si', 1)
DAVIDSON_SCREEN_THRESH = getattr (__config__, 'lassi_hsi_screen_thresh', 1e-12)
PSPACE_SIZE = getattr (__config__, 'lassi_hsi_pspace_size', 400)
PRIVREF = getattr (__config__, 'lassi_privref', True)

op = (op_o0, op_o1)

def kernel (sisolver, e0, h1, h2, norb_f, ci_fr, nelec_frs, smult_fr=None,
                disc_fr=None, soc=None, opt=None, max_memory=None, davidson_only=None):
    if soc is None:
        soc = getattr (sisolver, 'soc', 0)
    if opt is None:
        opt = getattr (sisolver, 'opt', 1)
    if davidson_only is None:
        davidson_only = getattr (sisolver, 'davidson_only', False)
    if max_memory is None:
        max_memory = getattr (sisolver, 'max_memory', param.MAX_MEMORY)
    nstates = np.prod (get_lroots (ci_fr), axis=0).sum ()
    req_memory = 24*nstates*nstates/1e6
    current_memory = lib.current_memory ()[0]
    sisolver.converged = False
    if current_memory+req_memory > max_memory:
        if opt==0:
            raise MemoryError ("Need %f MB of %f MB av (N.B.: o0 Davidson is fake; use opt=1)",
                               req_memory, max_memory-current_memory)
        logger.info (sisolver, ("Need %f MB of %f MB av for incore LASSI diag; Davidson alg "
                                    "forced"), req_memory, max_memory-current_memory)
    if davidson_only or current_memory+req_memory > max_memory:
        return kernel_Davidson (sisolver, e0, h1, h2, norb_f, ci_fr, nelec_frs, smult_fr,
                                    disc_fr, soc, opt)
    return kernel_incore (sisolver, e0, h1, h2, norb_f, ci_fr, nelec_frs, smult_fr, soc, opt)

def get_init_guess (sisolver, hdiag, nroots, si1, log=None, penalty=None):
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

class SISolver (lib.StreamObject):

    def __init__(self, las, soc=0, opt=1, davidson_only=False, nroots=NROOTS,
                 max_memory=param.MAX_MEMORY):
        self.las = las # I need this because op_o? fns need this
        self.verbose = las.verbose
        self.stdout = las.stdout
        self.max_memory = max_memory
        self.davidson_only = davidson_only
        self.level_shift = LEVEL_SHIFT
        self.davidson_screen_thresh = DAVIDSON_SCREEN_THRESH
        self.pspace_size = PSPACE_SIZE
        self.privref = PRIVREF
        self.conv_tol = CONV_TOL
        self.nroots = nroots
        self.smult_si = None
        self.converged = False
        self._keys = set((self.__dict__.keys()))

    kernel = kernel
    get_init_guess = get_init_guess

def kernel_Davidson (sisolver, e0, h1, h2, norb_f, ci_fr, nelec_frs, smult_fr, disc_fr, soc,
                         opt):
    # si0
    # nroots
    # level_shift
    verbose = sisolver.verbose
    davidson_log = log = logger.new_logger (sisolver, verbose)
    # We want this Davidson diagonalizer to be louder than usual
    if verbose >= logger.NOTE:
        davidson_log = logger.new_logger (sisolver, verbose+1)
    si0 = getattr (sisolver.las, 'si', None)
    level_shift = getattr (sisolver, 'level_shift', LEVEL_SHIFT)
    nroots = getattr (sisolver, 'nroots', NROOTS)
    max_cycle = getattr (sisolver, 'max_cycle', MAX_CYCLE)
    max_space = getattr (sisolver, 'max_space', MAX_SPACE)
    conv_tol = getattr (sisolver, 'conv_tol', CONV_TOL)
    privilege_ref = getattr (sisolver, 'privref', PRIVREF)
    screen_thresh = getattr (sisolver, 'davidson_screen_thresh', DAVIDSON_SCREEN_THRESH)
    pspace_size = getattr (sisolver, 'pspace_size', PSPACE_SIZE)
    smult_si = getattr (sisolver, 'smult_si', None)
    h_op_raw, s2_op, ovlp_op, hdiag_raw, _get_ovlp = op[opt].gen_contract_op_si_hdiag (
        sisolver.las, h1, h2, ci_fr, nelec_frs, smult_fr=smult_fr, soc=soc, disc_fr=disc_fr,
        screen_thresh=screen_thresh
    )
    if verbose >= logger.DEBUG:
        # The sort is slow
        log.debug ("fingerprint of hdiag raw: %15.10e", lib.fp (np.sort (hdiag_raw)))
    t0 = (logger.process_clock (), logger.perf_counter ())
    raw2orth = basis.get_orth_basis (ci_fr, norb_f, nelec_frs, _get_ovlp=_get_ovlp,
                                     smult_fr=smult_fr, smult_si=smult_si, disc_fr=disc_fr)
    raw2orth.log_debug1_hdiag_raw (log, hdiag_raw)
    orth2raw = raw2orth.H
    mem_orth = raw2orth.get_nbytes () / 1e6
    t0 = log.timer ('LASSI get orthogonal basis ({:.2f} MB)'.format (mem_orth), *t0)
    hdiag_orth = op[opt].get_hdiag_orth (hdiag_raw, h_op_raw, raw2orth)
    if verbose >= logger.DEBUG:
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
            pv = pv[:,:nroots]
            pw = pw[:nroots]
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
    x0 = sisolver.get_init_guess (hdiag_orth, nroots, x0, log=log, penalty=hdiag_penalty)
    def h_op (x):
        return raw2orth (h_op_raw (orth2raw (x)))
    log.info ("LASSI E(const) = %15.10f", e0)
    print ("right before davidson", nroots, flush=True)
    conv, e, x1 = lib.davidson1 (lambda xs: [h_op (x) for x in xs],
                                 x0, precond_op, nroots=nroots,
                                 verbose=davidson_log, max_cycle=max_cycle,
                                 max_space=max_space, tol=conv_tol)
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
    if (log is not None) and (log.verbose > logger.DEBUG) and (np.count_nonzero (idx_err)):
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

def kernel_incore (sisolver, e0, h1, h2, norb_f, ci_fr, nelec_frs, smult_fr, soc, opt):
    # TODO: simplify
    t0 = (logger.process_clock (), logger.perf_counter ())
     
    ham_blk, s2_blk, ovlp_blk, _get_ovlp = op[opt].ham (
        sisolver.las, h1, h2, ci_fr, nelec_frs, smult_fr=smult_fr, soc=soc)
    t0 = logger.timer (sisolver, 'LASSI H build', *t0)

    # Error catch: linear dependencies in basis
    raw2orth = basis.get_orth_basis (ci_fr, norb_f, nelec_frs, _get_ovlp=_get_ovlp,
                                     smult_fr=smult_fr)
    xhx = raw2orth (ham_blk.T).T
    logger.info (sisolver, '%d/%d linearly independent model states',
                     xhx.shape[1], xhx.shape[0])
    xhx = raw2orth (xhx.conj ()).conj ()
    try:
        e, c = linalg.eigh (xhx)
    except linalg.LinAlgError as err:
        ovlp_det = linalg.det (ovlp_blk)
        lc = 'checking if LASSI basis has lindeps: |ovlp| = {:.6e}'.format (ovlp_det)
        logger.info (sisolver, 'Caught error %s, %s', str (err), lc)
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

