import numpy as np
import time
from scipy import linalg
from mrh.my_pyscf.lassi import op_o0
from mrh.my_pyscf.lassi import op_o1
from mrh.my_pyscf.lassi import chkfile
from mrh.my_pyscf.lassi import citools
from mrh.my_pyscf.lassi.citools import get_lroots
from pyscf import lib, symm, ao2mo
from pyscf.scf.addons import canonical_orth_
from pyscf.lib.numpy_helper import tag_array
from pyscf.fci.direct_spin1 import _unpack_nelec
from itertools import combinations, product
from mrh.my_pyscf.mcscf import soc_int as soc_int
from pyscf import __config__
from mrh.my_pyscf.lassi.spaces import list_spaces

# TODO: fix stdm1 index convention in both o0 and o1

# TODO: adopt consistent nomenclature viz "states", "spaces", "roots"

# TODO: remove the dependence of lassi_op_o1 on las.fciboxes in some way
# The fcisolvers contain linkstr and symmetry information, but probably
# only the former is necessary. Once the connection to the parent LAS
# instance is severed, remove the dangerous "_LASSI_subspace_env"
# temporary environment.

LINDEP_THRESH = getattr (__config__, 'lassi_lindep_thresh', 1.0e-5)
LEVEL_SHIFT_SI = getattr (__config__, 'lassi_level_shift_si', 1.0e-8)
NROOTS_SI = 1

op = (op_o0, op_o1)

def ham_2q (las, mo_coeff, veff_c=None, h2eff_sub=None, soc=0):
    '''Construct second-quantization Hamiltonian in CAS, using intermediates from
    a LASSCF calculation.

    Args:
        las : instance of :class:`LASCINoSymm`
        mo_coeff: ndarray of shape (nao,nmo)
            Contains MO coefficients

    Kwargs:
        veff_c : ndarray of shape (nao,nao)
            Effective potential of inactive electrons
        h2eff_sub : ndarray of shape (nmo,(ncas**2)*(ncas+1)/2)
            Two-electron integrals
        soc : integer
            Order of spin-orbit coupling to include. Currently only 0 or 1 supported.
            Including spin-orbit coupling increases the size of return arrays to
            account for additional spin-symmetry-breaking sectors of the Hamiltonian.

    Returns:
        h0 : float
            Constant part of the CAS Hamiltonian
        h1 : ndarray of shape (ncas,ncas) or (2*ncas,2*ncas)
            One-electron part of the CAS Hamiltonian. If soc==True, h1 is returned in
            the spinorbital representation: ncas spin-up states followed by ncas
            spin-down states
        h2 : ndarray of shape [ncas,]*4
            Two-electron part of the CAS Hamiltonian.
    '''
    # a'b = sx + i*sy
    # b'a = sx - i*sy
    # a'a = 1/2 n + sz
    # b'b = 1/2 n - sz
    #   ->
    # sx =  (1/2) (a'b + b'a)
    # sy = (-i/2) (a'b - b'a)
    # sz =  (1/2) (a'a - b'b)
    # 
    # l.s = lx.sx + ly.sy + lz.sz
    #     = (1/2) (lx.(a'b + b'a) - i*ly.(a'b - b'a) + lz.(a'a - b'b))
    #     = (1/2) (  (lx - i*ly).a'b 
    #              + (lx + i*ly).b'a
    #              + lz.(a'a - b'b)
    #             )
    if isinstance (las, LASSI): las = las._las
    if soc>1: raise NotImplementedError ("Two-electron spin-orbit coupling")
    ncore, ncas, nocc = las.ncore, las.ncas, las.ncore + las.ncas
    norb = sum(las.ncas_sub)
    mo_core = mo_coeff[:,:ncore]
    mo_cas = mo_coeff[:,ncore:nocc]
    hcore = las._scf.get_hcore ()
    if veff_c is None: 
        dm_core = 2 * mo_core @ mo_core.conj ().T
        veff_c = las.get_veff (dm=dm_core)

    h0 = las._scf.energy_nuc () + 2 * (((hcore + veff_c/2) @ mo_core) * mo_core).sum ()

    h1 = mo_cas.conj ().T @ (hcore + veff_c) @ mo_cas
    if soc:
        dm0 = soc_int.amfi_dm (las.mol)
        hsoao = soc_int.compute_hso(las.mol, dm0, amfi=True)
        hso = .5*lib.einsum ('ip,rij,jq->rpq', mo_cas.conj (), hsoao, mo_cas)

        h1 = linalg.block_diag (h1, h1).astype (complex)
        h1[ncas:2*ncas,0:ncas] = (hso[0] + 1j * hso[1]) # b'a
        h1[0:ncas,ncas:2*ncas] = (hso[0] - 1j * hso[1]) # a'b
        h1[0:ncas,0:ncas] += hso[2] # a'a
        h1[ncas:2*ncas,ncas:2*ncas] -= hso[2] # b'b

    if h2eff_sub is None:
        h2 = las.get_h2cas (mo_coeff)
    else:
        h2 = h2eff_sub[ncore:nocc].reshape (ncas*ncas, ncas * (ncas+1) // 2)
        h2 = lib.numpy_helper.unpack_tril (h2).reshape (ncas, ncas, ncas, ncas)

    return h0, h1, h2

def las_symm_tuple (las, spaces=None, break_spin=False, break_symmetry=False, verbose=None):
    '''Identify the symmetries/quantum numbers of of each LAS excitation space within a LASSI
    model space which are to be preserved by projecting the Hamiltonian into the corresponding
    diagonal symmetry blocks.

    Args:
        las : instance of :class:`LASCINoSymm`

    Kwargs:
        spaces : list of instances of :class:`SingleLASRootspace`
            Contain symmetry information
        break_spin : logical
            Whether to mix states of different neleca-nelecb (necessary for spin-orbit coupling).
            If True, the first item of each element in statesym is the total number of electrons;
            otherwise, the first two items are the total number of spin-up and spin-down
            electrons.
        break_symmetry : logical
            Whether to mix states of different point-group irreps (may also be necessary for
            spin-orbit coupling). If True, the point-group irrep of each state is omitted from
            the elements of statesym; otherwise, this datum is the last item of each element.

    Returns:
        statesym : list of length nroots
            Each element is a tuple describing all enforced symmetries of a LAS state. 
            The length of each tuple varies between 1 and 4 based on the kwargs break_spin and
            break_symmetry.
        s2_states : list of length nroots
            The expectation values of the <S**2> operator for each state, for convenience
    '''
    if spaces is None: spaces = list_spaces (las)
    # kwarg logic setup
    qn_lbls = ['Neleca', 'Nelecb', 'Nelec', 'Wfnsym']
    incl_spin = not (bool (break_spin))
    incl_symmetry = not (bool (break_symmetry))
    qn_filter = [incl_spin, incl_spin, not incl_spin, incl_symmetry]
    # end kwarg logic setup
    full_statesym = [] # keep everything for i/o...
    statesym = [] # ...but return only this
    s2_states = []
    log = lib.logger.new_logger (las, verbose)
    for space in spaces:
        s2_states.append (space.get_s2_exptval ())
        neleca = space.neleca.sum ()
        nelecb = space.nelecb.sum ()
        wfnsym = space.get_wfnsym ()
        all_qns = [neleca, nelecb, neleca+nelecb, wfnsym]
        full_statesym.append (tuple (all_qns))
        statesym.append (tuple (qn for qn, incl in zip (all_qns, qn_filter) if incl))
    log.info ('Symmetry analysis of %d LAS rootspaces:', las.nroots)
    qn_lbls = ['Neleca', 'Nelecb', 'Nelec', 'Wfnsym']
    qn_fmts = ['{:6d}', '{:6d}', '{:6d}', '{:>6s}']
    lbls = ['ix', 'Energy', '<S**2>'] + qn_lbls
    fmt_str = ' {:2s}  {:>16s}  {:6s}  ' + '  '.join (['{:6s}',]*len(qn_lbls))
    log.info (fmt_str.format (*lbls))
    try:
        for ix, (e, sy, s2) in enumerate (zip (las.e_states, full_statesym, s2_states)):
            data = [ix, e, s2] + list (sy)
            data[-1] = symm.irrep_id2name (las.mol.groupname, data[-1])
            fmts = ['{:2d}','{:16.10f}','{:6.3f}'] + qn_fmts
            fmt_str = ' ' + '  '.join (fmts)
            log.info (fmt_str.format (*data))
    except TypeError as err:
        print (las.e_states, full_statesym, s2_states)
        raise (err)
    if break_spin:
        log.info ("States with different neleca-nelecb can be mixed by LASSI")
    if break_symmetry:
        log.info ("States with different point-group symmetry can be mixed by LASSI")

    return statesym, np.asarray (s2_states)

class _LASSI_subspace_env (object):
    def __init__(self, las, my_fcisolvers, my_e_states):
        self.las = las
        self.my_fcisolvers = my_fcisolvers
        self.my_e_states = my_e_states
        self.fcisolvers = [f.fcisolvers for f in las.fciboxes]
        self.e_states = las.e_states
    def __enter__(self):
        for f, g in zip (self.las.fciboxes, self.my_fcisolvers):
            f.fcisolvers = g
        self.las.e_states = self.my_e_states
    def __exit__(self, type, value, traceback):
        for ix, f in enumerate (self.las.fciboxes):
            f.fcisolvers = self.fcisolvers[ix]
        self.las.e_states = self.e_states

def iterate_subspace_blocks (las, ci, spacesym, subset=None, spaces=None):
    if subset is None: subset = set (spacesym)
    lroots = get_lroots (ci)
    nprods_r = np.prod (lroots, axis=0)
    prod_off = np.cumsum (nprods_r) - nprods_r
    nprods = nprods_r.sum ()
    if spaces is None: spaces = list_spaces (las)
    for sym in subset:
        idx_space = np.all (np.array (spacesym) == sym, axis=1)
        idx = np.where (idx_space)[0]
        ci_blk = [[c[i] for i in idx] for c in ci]
        idx_prod = np.zeros (nprods, dtype=bool)
        my_fcisolvers = [[] for i in range (las.nfrags)]
        my_e_states = []
        nelec_blk = np.zeros ((las.nfrags,len(idx),2), dtype=int)
        for i0, i in enumerate (idx):
            idx_prod[prod_off[i]:prod_off[i]+nprods_r[i]] = True
            my_fcisolvers_i = spaces[i].get_fcisolvers ()
            nelec_blk[:,i0,:] = np.stack ([spaces[i].neleca, spaces[i].nelecb], axis=1)
            for j in range (las.nfrags):
                my_fcisolvers[j].append (my_fcisolvers_i[j])
            my_e_states.append (spaces[i].energy_tot)
        with _LASSI_subspace_env (las, my_fcisolvers, my_e_states):
            yield las, sym, (idx_space, idx_prod), (ci_blk, nelec_blk)

class LASSIOop01DisagreementError (RuntimeError):
    def __init__(self, message, errvec):
        self.message = message + ("\n"
            "max abs errvec = {}; ||errvec|| = {}").format (
                np.amax (np.abs (errvec)), linalg.norm (errvec))
        self.errvec = errvec
    def __str__(self):
        return self.message

def lassi (las, mo_coeff=None, ci=None, veff_c=None, h2eff_sub=None, orbsym=None, soc=False,
           break_symmetry=False, opt=1, davidson_only=None):
    ''' Diagonalize the state-interaction matrix of LASSCF '''
    if mo_coeff is None: mo_coeff = las.mo_coeff
    if ci is None: ci = las.ci
    if orbsym is None: 
        orbsym = getattr (las.mo_coeff, 'orbsym', None)
        if orbsym is None and callable (getattr (las, 'label_symmetry_', None)):
            orbsym = las.label_symmetry_(las.mo_coeff).orbsym
        if orbsym is not None:
            orbsym = orbsym[las.ncore:las.ncore+las.ncas]
    if davidson_only is None: davidson_only = getattr (las, 'davidson_only', False)
    max_memory = getattr (las, 'max_memory', 2000)
    o0_memcheck = op_o0.memcheck (las, ci, soc=soc)
    if opt == 0 and o0_memcheck == False:
        raise RuntimeError ('Insufficient memory to use o0 LASSI algorithm')

    # Construct second-quantization Hamiltonian
    if callable (getattr (las, 'ham_2q', None)):
        e0, h1, h2 = las.ham_2q (mo_coeff, veff_c=veff_c, h2eff_sub=h2eff_sub, soc=soc)
    else:
        e0, h1, h2 = ham_2q (las, mo_coeff, veff_c=veff_c, h2eff_sub=h2eff_sub, soc=soc)

    # Symmetry tuple: neleca, nelecb, irrep
    statesym, s2_states = las_symm_tuple (las, break_spin=soc, break_symmetry=break_symmetry)

    # Initialize matrices
    e_roots = []
    s2_roots = []
    rootsym = []
    si = []
    idx_allprods = []
    dtype = complex if soc else np.float64

    # Loop over symmetry blocks
    qn_lbls = ['nelec',] if soc else ['neleca','nelecb',]
    if not break_symmetry: qn_lbls.append ('irrep')
    for it, (las1,sym,indices,indexed) in enumerate (iterate_subspace_blocks(las,ci,statesym)):
        idx_space, idx_prod = indices
        ci_blk, nelec_blk = indexed
        idx_allprods.extend (list(np.where(idx_prod)[0]))
        lib.logger.info (las, 'Build + diag H matrix LASSI symmetry block %d\n'
                         + '{} = {}\n'.format (qn_lbls, sym)
                         + '(%d rootspaces; %d states)', it,
                         np.count_nonzero (idx_space), 
                         np.count_nonzero (idx_prod))
        if np.count_nonzero (idx_prod) == 1:
            lib.logger.debug (las, 'Only one state in this symmetry block')
            e_roots.extend (las1.e_states - e0)
            si.append (np.ones ((1,1), dtype=dtype))
            s2_roots.extend (s2_states[idx_space])
            rootsym.extend ([sym,])
            continue
        wfnsym = None if break_symmetry else sym[-1]
        las.converged_si, e, c, s2_blk = _eig_block (las1, e0, h1, h2, ci_blk, nelec_blk, soc, opt,
                                                     davidson_only=davidson_only,
                                                     max_memory=max_memory)
        si.append (c)
        e_roots.extend (list(e))
        s2_roots.extend (list (s2_blk))
        rootsym.extend ([sym,]*c.shape[1])

    # The matrix blocks were evaluated in idx_allprods order
    # Therefore, I need to ~invert~ idx_allprods to get the proper order
    idx_allprods = np.argsort (idx_allprods)
    si = linalg.block_diag (*si)[idx_allprods,:]

    # Sort results by energy
    idx = np.argsort (e_roots)
    rootsym = np.asarray (rootsym)[idx]
    e_roots = np.asarray (e_roots)[idx] + e0
    s2_roots = np.asarray (s2_roots)[idx]
    if soc == False:
        nelec_roots = [tuple(rs[0:2]) for rs in rootsym]
    else:
        nelec_roots = [rs[0] for rs in rootsym]
    if break_symmetry:
        wfnsym_roots = [None for rs in rootsym]
    else:
        wfnsym_roots = [rs[-1] for rs in rootsym]

    # Results tagged on si array....
    si = si[:,idx]
    si = tag_array (si, s2=s2_roots, nelec=nelec_roots, wfnsym=wfnsym_roots,
                    rootsym=rootsym, break_symmetry=break_symmetry, soc=soc)

    # I/O
    lib.logger.info (las, 'LASSI eigenvalues (%d total):', len (e_roots))
    fmt_str = ' {:2s}  {:>16s}  {:6s}  '
    col_lbls = ['Nelec'] if soc else ['Neleca','Nelecb']
    if not break_symmetry: col_lbls.append ('Wfnsym')
    fmt_str += '  '.join (['{:6s}',]*len(col_lbls))
    col_lbls = ['ix','Energy','<S**2>'] + col_lbls
    lib.logger.info (las, fmt_str.format (*col_lbls))
    fmt_str = ' {:2d}  {:16.10f}  {:6.3f}  '
    col_fmts = ['{:6d}',]*(2-int(soc))
    if not break_symmetry: col_fmts.append ('{:>6s}')
    fmt_str += '  '.join (col_fmts)
    for ix, (er, s2r, rsym) in enumerate (zip (e_roots, s2_roots, rootsym)):
        if np.iscomplexobj (s2r):
            assert (abs (s2r.imag) < 1e-8)
            s2r = s2r.real
        nelec = rsym[0:1] if soc else rsym[:2]
        row = [ix,er,s2r] + list (nelec)
        if not break_symmetry: row.append (symm.irrep_id2name (las.mol.groupname, rsym[-1]))
        lib.logger.info (las, fmt_str.format (*row))
        if ix>=99 and las.verbose < lib.logger.DEBUG:
            lib.logger.info (las, ('Remaining %d eigenvalues truncated; '
                                   'increase verbosity to print them all'), len (e_roots)-100)
            break
    return e_roots, si

def _eig_block (las, e0, h1, h2, ci_blk, nelec_blk, soc, opt, max_memory=2000,
                davidson_only=False):
    nstates = np.prod (get_lroots (ci_blk), axis=0).sum ()
    req_memory = 24*nstates*nstates/1e6
    current_memory = lib.current_memory ()[0]
    if current_memory+req_memory > max_memory:
        # TODO: Efficient LASSI op_o2 h_op sivec 
        raise MemoryError ("Need %f MB of %f MB avail (N.B.: Davidson implementation is fake)",
                           req_memory, max_memory-current_memory)
        lib.logger.info ("Need %f MB of %f MB avail for incore LASSI diag; Davidson alg forced",
                         req_memory, max_memory-current_memory)
    if davidson_only or current_memory+req_memory > max_memory:
        return _eig_block_Davidson (las, e0, h1, h2, ci_blk, nelec_blk, soc, opt)
    return _eig_block_incore (las, e0, h1, h2, ci_blk, nelec_blk, soc, opt)

def _eig_block_Davidson (las, e0, h1, h2, ci_blk, nelec_blk, soc, opt):
    # si0
    # nroots_si
    # level_shift
    si0 = getattr (las, 'si', None)
    level_shift = getattr (las, 'level_shift_si', LEVEL_SHIFT_SI)
    nroots_si = getattr (las, 'nroots_si', NROOTS_SI)
    get_init_guess = getattr (las, 'get_init_guess_si', get_init_guess_si)
    h_op_raw, s2_op, ovlp_op, hdiag, _get_ovlp = op[opt].gen_contract_op_si_hdiag (
        las, h1, h2, ci_blk, nelec_blk, soc=soc
    )
    raw2orth = citools.get_orth_basis (ci_blk, las.ncas_sub, nelec_blk, _get_ovlp=_get_ovlp)
    precond_op_raw = lib.make_diag_precond (hdiag, level_shift=level_shift)
    si0 = get_init_guess (hdiag, nroots_si, si0)
    orth2raw = raw2orth.H
    def precond_op (dx, e, *args):
        return raw2orth (precond_op_raw (orth2raw (dx), e, *args))
    def h_op (x):
        return raw2orth (h_op_raw (orth2raw (x)))
    x0 = [raw2orth (x) for x in si0]
    conv, e, x1 = lib.davidson1 (lambda xs: [h_op (x) for x in xs],
                                  x0, precond_op, nroots=nroots_si)
    conv = all (conv)
    si1 = np.stack ([orth2raw (x) for x in x1], axis=-1)
    s2 = np.array ([np.dot (x.conj (), s2_op (x)) for x in si1.T])
    return conv, e, si1, s2

def get_init_guess_si (hdiag, nroots, si1):
    nprod = hdiag.size
    si0 = []
    if nprod <= nroots:
        addrs = np.arange(nprod)
    else:
        addrs = np.argpartition(hdiag, nroots-1)[:nroots]
    for addr in addrs:
        x = np.zeros((nprod))
        x[addr] = 1
        si0.append(x)
    # Add noise
    si0[0][0 ] += 1e-5
    si0[0][-1] -= 1e-5
    if si1 is not None:
        si1 = si1.reshape (nprod,-1)
        for i in range (min (si1.shape[1], nroots)):
            si0[i] = si1[:,i]
    return si0

def _eig_block_incore (las, e0, h1, h2, ci_blk, nelec_blk, soc, opt):
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
            las, h1, h2, ci_blk, nelec_blk, soc=soc)
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
    raw2orth = citools.get_orth_basis (ci_blk, las.ncas_sub, nelec_blk, _get_ovlp=_get_ovlp)
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

def make_stdm12s (las, ci=None, orbsym=None, soc=False, break_symmetry=False, spaces=None, opt=1):
    ''' Evaluate <I|p'q|J> and <I|p'r'sq|J> where |I>, |J> are LAS states.

        Args:
            las: LASCI object

        Kwargs:
            ci: list of list of ci vectors
            orbsym: None or list of orbital symmetries spanning the whole orbital space
            soc: logical
                Whether to include the effects of spin-orbit coupling (in the 1-RDMs only)
            break_symmetry: logical
                Whether to allow coupling between states of different point-group irreps
            spaces : list of instances of :class:`SingleLASRootspace`
                Contain symmetry information; defaults to data from las
            opt: Optimization level, i.e.,  take outer product of
                0: CI vectors
                1: TDMs

        Returns:
            stdm1s: ndarray of shape (nroots,2,ncas,ncas,nroots) if soc==False;
                or of shape (nroots,2*ncas,2*ncas,nroots) if soc==True.
            stdm2s: ndarray of shape (nroots,2,ncas,ncas,2,ncas,ncas,nroots)
    '''
    # NOTE: A spin-pure dm1s is two ncas-by-ncas matrices,
    #    _______    _______
    #    |     |    |     |
    #  [ | a'a | ,  | b'b | ]
    #    |     |    |     |
    #    -------    -------  
    # Spin-orbit coupling generates the a'b and b'a sectors, which are
    # in the missing off-diagonal blocks,
    # _____________
    # |     |     |  
    # | a'a | a'b |  
    # |     |     |  
    # -------------
    # |     |     |  
    # | b'a | b'b |  
    # |     |     |  
    # -------------

    if ci is None: ci = las.ci
    if orbsym is None: 
        orbsym = getattr (las.mo_coeff, 'orbsym', None)
        if orbsym is None and callable (getattr (las, 'label_symmetry_', None)):
            orbsym = las.label_symmetry_(las.mo_coeff).orbsym
        if orbsym is not None:
            orbsym = orbsym[las.ncore:las.ncore+las.ncas]
    o0_memcheck = op_o0.memcheck (las, ci, soc=soc)
    if opt == 0 and o0_memcheck == False:
        raise RuntimeError ('Insufficient memory to use o0 LASSI algorithm')

    # Loop over symmetry blocks
    statesym = las_symm_tuple (las, spaces=spaces, break_spin=soc, break_symmetry=break_symmetry,
                               verbose=0)[0]
    idx_allprods = []
    d1s_all = []
    d2s_all = []
    nprods = 0
    for las1, sym, indices, indexed in iterate_subspace_blocks (las, ci, statesym, spaces=spaces):
        idx_sp, idx_prod = indices
        ci_blk, nelec_blk = indexed
        t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
        wfnsym = None if break_symmetry else sym[-1]
        # TODO: implement SOC in op_o1 and then re-enable the debugging block below
        if (las.verbose > lib.logger.INFO) and (o0_memcheck) and (soc==False):
            d1s, d2s = op_o0.make_stdm12s (las1, ci_blk, nelec_blk, orbsym=orbsym, wfnsym=wfnsym)
            t0 = lib.logger.timer (las, 'LASSI make_stdm12s rootsym {} CI algorithm'.format (
                sym), *t0)
            d1s_test, d2s_test = op_o1.make_stdm12s (las1, ci_blk, nelec_blk)
            t0 = lib.logger.timer (las, 'LASSI make_stdm12s rootsym {} TDM algorithm'.format (
                sym), *t0)
            lib.logger.debug (las,
                'LASSI make_stdm12s rootsym {}: D1 o0-o1 algorithm disagreement = {}'.format (
                    sym, linalg.norm (d1s_test - d1s))) 
            lib.logger.debug (las,
                'LASSI make_stdm12s rootsym {}: D2 o0-o1 algorithm disagreement = {}'.format (
                    sym, linalg.norm (d2s_test - d2s))) 
            errvec = np.concatenate ([(d1s-d1s_test).ravel (), (d2s-d2s_test).ravel ()])
            if np.amax (np.abs (errvec)) > 1e-8:#
                raise LASSIOop01DisagreementError ("State-transition density matrices", errvec)
            if opt == 1:
                d1s = d1s_test
                d2s = d2s_test
        else:
            if not o0_memcheck: lib.logger.debug (
                las, 'Insufficient memory to test against o0 LASSI algorithm')
            d1s, d2s = op[opt].make_stdm12s (las1, ci_blk, nelec_blk, orbsym=orbsym, wfnsym=wfnsym)
            t0 = lib.logger.timer (las, 'LASSI make_stdm12s rootsym {}'.format (sym), *t0)
        idx_allprods.append (list(np.where(idx_prod)[0]))
        nprods += len (idx_allprods[-1])
        d1s_all.append (d1s)
        d2s_all.append (d2s)

    # Sort block-diagonal matrices
    norb = las.ncas
    if soc:
        stdm1s = np.zeros ((nprods, nprods, 2*norb, 2*norb),
            dtype=ci[0][0].dtype).transpose (0,2,3,1)
    else:
        stdm1s = np.zeros ((nprods, nprods, 2, norb, norb),
            dtype=ci[0][0].dtype).transpose (0,2,3,4,1)
    # TODO: 2e- SOC
    stdm2s = np.zeros ((nprods, nprods, 2, norb, norb, 2, norb, norb),
        dtype=ci[0][0].dtype).transpose (0,2,3,4,5,6,7,1)
    for idx_prod, d1s, d2s in zip (idx_allprods, d1s_all, d2s_all):
        for (i,a), (j,b) in product (enumerate (idx_prod), repeat=2):
            stdm1s[a,...,b] = d1s[i,...,j]
            stdm2s[a,...,b] = d2s[i,...,j]
    return stdm1s, stdm2s

def guess_rootsym (si, statesym, lroots):
    rootsym = []
    nprods = np.prod (lroots, axis=0)
    offs1 = np.cumsum (nprods)
    offs0 = offs1 - nprods
    for sivec in si.T:
        # TODO: refactor rootsym and statesym and eliminate this kludge
        sivec = sivec[:offs1[len(statesym)-1]]
        idx = np.argmax (np.abs (sivec))
        iroot = np.where (np.logical_and (idx>=offs0, idx<offs1))[0][0]
        rootsym.append (statesym[iroot])
    return rootsym

def roots_trans_rdm12s (las, ci, si_bra, si_ket, orbsym=None, soc=None, break_symmetry=None,
                        spaces=None, opt=1):
    '''Evaluate 1- and 2-electron reduced transition density matrices of LASSI states

        Args:
            las: LASCI object
            ci: list of list of ci vectors
            si_bra: tagged ndarray of shape (nroots,nroots)
               Linear combination vectors defining LASSI states for the bra.
            si_ket: tagged ndarray of shape (nroots,nroots)
               Linear combination vectors defining LASSI states for the ket.

        Kwargs:
            orbsym: None or list of orbital symmetries spanning the whole orbital space
            soc: logical
                Whether to include the effects of spin-orbit coupling (in the 1-RDMs only)
                Overrides tag of si if provided by caller. I have no idea what will happen
                if they contradict. This should probably be removed.
            break_symmetry: logical
                Whether to allow coupling between states of different point-group irreps
                Overrides tag of si if provided by caller. I have no idea what will happen
                if they contradict. This should probably be removed.
            spaces : list of instances of :class:`SingleLASRootspace`
                Contain symmetry information; defaults to data from las
            opt: Optimization level, i.e.,  take outer product of
                0: CI vectors
                1: TDMs

        Returns:
            rdm1s: ndarray of shape (nroots,2,ncas,ncas) if soc==False;
                or of shape (nroots,2*ncas,2*ncas) if soc==True.
            rdm2s: ndarray of shape (nroots,2,ncas,ncas,2,ncas,ncas)
    '''
    if orbsym is None: 
        orbsym = getattr (las.mo_coeff, 'orbsym', None)
        if orbsym is None and callable (getattr (las, 'label_symmetry_', None)):
            orbsym = las.label_symmetry_(las.mo_coeff).orbsym
        if orbsym is not None:
            orbsym = orbsym[las.ncore:las.ncore+las.ncas]
    if soc is None:
        soc = getattr (si_ket, 'soc', getattr (las, 'soc', False))
    if break_symmetry is None:
        break_symmetry = getattr (si_ket, 'break_symmetry', getattr (las, 'break_symmetry', False))
    o0_memcheck = op_o0.memcheck (las, ci, soc=soc)
    if opt == 0 and o0_memcheck == False:
        raise RuntimeError ('Insufficient memory to use o0 LASSI algorithm')

    # Initialize matrices
    norb = las.ncas
    nroots = si_ket.shape[1]
    rdm1s = [None for i in range (nroots)]
    rdm2s = [None for i in range (nroots)]
    #if soc:
    #    rdm1s = np.zeros ((nroots, 2*norb, 2*norb),
    #        dtype=si.dtype)
    #else:
    #    rdm1s = np.zeros ((nroots, 2, norb, norb),
    #        dtype=si.dtype)
    ## TODO: 2e- SOC
    #rdm2s = np.zeros ((nroots, 2, norb, norb, 2, norb, norb),
    #    dtype=si.dtype)

    # Loop over symmetry blocks
    statesym = las_symm_tuple (las, spaces=spaces, break_spin=soc, break_symmetry=break_symmetry,
                               verbose=0)[0]
    lroots = get_lroots (ci)
    rootsym = guess_rootsym (si_bra, statesym, lroots)
    rootsym_ket = guess_rootsym (si_ket, statesym, lroots)
    assert (all ([b==k for b, k in zip (rootsym, rootsym_ket)]))
    for las1, sym, indcs, indxd in iterate_subspace_blocks(las,ci,statesym,subset=set(rootsym),spaces=spaces):
        idx_ci, idx_prod = indcs
        ci_blk, nelec_blk = indxd
        idx_si = np.all (np.array (rootsym) == sym, axis=1)
        wfnsym = None if break_symmetry else sym[-1]
        sib_blk = si_bra[np.ix_(idx_prod,idx_si)]
        sik_blk = si_ket[np.ix_(idx_prod,idx_si)]
        t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
        # TODO: implement SOC in op_o1 and then re-enable the debugging block below
        if (las.verbose > lib.logger.INFO) and (o0_memcheck) and (soc==False):
            d1s, d2s = op_o0.roots_trans_rdm12s (las1, ci_blk, nelec_blk, sib_blk, sik_blk,
                                                 orbsym=orbsym, wfnsym=wfnsym)
            t0 = lib.logger.timer (las, 'LASSI trans_rdm12s rootsym {} CI algorithm'.format (sym),
                                   *t0)
            d1s_test, d2s_test = op[opt].roots_trans_rdm12s (las1, ci_blk, nelec_blk, sib_blk,
                                                             sik_blk)
            t0 = lib.logger.timer (las, 'LASSI trans_rdm12s rootsym {} TDM algorithm'.format (sym),
                                   *t0)
            lib.logger.debug (las,
                'LASSI trans_rdm12s rootsym {}: D1 o0-o1 algorithm disagreement = {}'.format (
                    sym, linalg.norm (d1s_test - d1s))) 
            lib.logger.debug (las,
                'LASSI trans_rdm12s rootsym {}: D2 o0-o1 algorithm disagreement = {}'.format (
                    sym, linalg.norm (d2s_test - d2s))) 
            errvec = np.concatenate ([(d1s-d1s_test).ravel (), (d2s-d2s_test).ravel ()])
            if np.amax (np.abs (errvec)) > 1e-8 and soc == False: # tmp until SOC in for op_o1
                raise LASSIOop01DisagreementError ("LASSI mixed-state RDMs", errvec)
            if opt > 0:
                d1s = d1s_test
                d2s = d2s_test
        else:
            if not o0_memcheck: lib.logger.debug (las,
                'Insufficient memory to test against o0 LASSI algorithm')
            d1s, d2s = op[opt].roots_trans_rdm12s (las1, ci_blk, nelec_blk, sib_blk, sik_blk,
                                                   orbsym=orbsym, wfnsym=wfnsym)
            t0 = lib.logger.timer (las, 'LASSI trans_rdm12s rootsym {}'.format (sym), *t0)
        idx_int = np.where (idx_si)[0]
        for (i,a) in enumerate (idx_int):
            rdm1s[a] = d1s[i]
            rdm2s[a] = d2s[i]
    rdm1s = np.stack (rdm1s, axis=0)
    rdm2s = np.stack (rdm2s, axis=0)
    return rdm1s, rdm2s

def roots_make_rdm12s (las, ci, si, orbsym=None, soc=None, break_symmetry=None,
                       spaces=None, opt=1):
    '''Evaluate 1- and 2-electron reduced density matrices of LASSI states

        Args:
            las: LASCI object
            ci: list of list of ci vectors
            si: tagged ndarray of shape (nroots,nroots)
               Linear combination vectors defining LASSI states.

        Kwargs:
            orbsym: None or list of orbital symmetries spanning the whole orbital space
            soc: logical
                Whether to include the effects of spin-orbit coupling (in the 1-RDMs only)
                Overrides tag of si if provided by caller. I have no idea what will happen
                if they contradict. This should probably be removed.
            break_symmetry: logical
                Whether to allow coupling between states of different point-group irreps
                Overrides tag of si if provided by caller. I have no idea what will happen
                if they contradict. This should probably be removed.
            spaces : list of instances of :class:`SingleLASRootspace`
                Contain symmetry information; defaults to data from las
            opt: Optimization level, i.e.,  take outer product of
                0: CI vectors
                1: TDMs

        Returns:
            rdm1s: ndarray of shape (nroots,2,ncas,ncas) if soc==False;
                or of shape (nroots,2*ncas,2*ncas) if soc==True.
            rdm2s: ndarray of shape (nroots,2,ncas,ncas,2,ncas,ncas)
    '''
    return roots_trans_rdm12s (las, ci, si, si, orbsym=orbsym, soc=soc,
                               break_symmetry=break_symmetry, spaces=spaces, opt=opt)

def root_trans_rdm12s (las, ci, si_bra, si_ket, state=0, orbsym=None, soc=None, break_symmetry=None,
                       spaces=None, opt=1):
    '''Evaluate 1- and 2-electron reduced transition density matrices of one single pair of LASSI
    states.

        Args:
            las: LASCI object
            ci: list of list of ci vectors
            si_bra: tagged ndarray of shape (nroots,nroots)
               Linear combination vectors defining LASSI states for the bra.
            si_ket: tagged ndarray of shape (nroots,nroots)
               Linear combination vectors defining LASSI states for the ket.

        Kwargs:
            state: integer or sequence of integers
                Identify the specific LASSI eigenstate(s) for which the density matrices are
                to be computed.
            orbsym: None or list of orbital symmetries spanning the whole orbital space
            soc: logical
                Whether to include the effects of spin-orbit coupling (in the 1-RDMs only)
                Overrides tag of si if provided by caller. I have no idea what will happen
                if they contradict. This should probably be removed.
            break_symmetry: logical
                Whether to allow coupling between states of different point-group irreps
                Overrides tag of si if provided by caller. I have no idea what will happen
                if they contradict. This should probably be removed.
            spaces : list of instances of :class:`SingleLASRootspace`
                Contain symmetry information; defaults to data from las
            opt: Optimization level, i.e.,  take outer product of
                0: CI vectors
                1: TDMs

        Returns:
            rdm1s: ndarray of shape (2,ncas,ncas) if soc==False;
                or of shape (2*ncas,2*ncas) if soc==True.
            rdm2s: ndarray of shape (2,ncas,ncas,2,ncas,ncas)
    '''
    states = np.atleast_1d (state)
    sib_column = si_bra[:,states]
    sik_column = si_ket[:,states]
    if soc is None:
        soc = getattr (si_ket, 'soc', getattr (las, 'soc', False))
    if break_symmetry is None:
        break_symmetry = getattr (si_ket, 'break_symmetry', getattr (las, 'break_symmetry', False))
    rdm1s, rdm2s = roots_trans_rdm12s (las, ci, sib_column, sik_column, orbsym=orbsym, soc=soc,
                                       break_symmetry=break_symmetry, spaces=spaces, opt=opt)
    if len (states) == 1:
        rdm1s, rdm2s = rdm1s[0], rdm2s[0]
    return rdm1s, rdm2s

def root_make_rdm12s (las, ci, si, state=0, orbsym=None, soc=None, break_symmetry=None,
                      spaces=None, opt=1):
    '''Evaluate 1- and 2-electron reduced density matrices of one single LASSI state

        Args:
            las: LASCI object
            ci: list of list of ci vectors
            si: tagged ndarray of shape (nroots,nroots)
               Linear combination vectors defining LASSI states.

        Kwargs:
            state: integer or sequence of integers
                Identify the specific LASSI eigenstate(s) for which the density matrices are
                to be computed.
            orbsym: None or list of orbital symmetries spanning the whole orbital space
            soc: logical
                Whether to include the effects of spin-orbit coupling (in the 1-RDMs only)
                Overrides tag of si if provided by caller. I have no idea what will happen
                if they contradict. This should probably be removed.
            break_symmetry: logical
                Whether to allow coupling between states of different point-group irreps
                Overrides tag of si if provided by caller. I have no idea what will happen
                if they contradict. This should probably be removed.
            spaces : list of instances of :class:`SingleLASRootspace`
                Contain symmetry information; defaults to data from las
            opt: Optimization level, i.e.,  take outer product of
                0: CI vectors
                1: TDMs

        Returns:
            rdm1s: ndarray of shape (2,ncas,ncas) if soc==False;
                or of shape (2*ncas,2*ncas) if soc==True.
            rdm2s: ndarray of shape (2,ncas,ncas,2,ncas,ncas)
    '''
    return root_trans_rdm12s (las, ci, si, si, state=state, orbsym=orbsym, soc=soc,
                              break_symmetry=break_symmetry, spaces=spaces, opt=opt)

def energy_tot (lsi, mo_coeff=None, ci=None, si=None, soc=0, opt=None):
    if mo_coeff is None: mo_coeff = lsi.mo_coeff
    if ci is None: ci = lsi.ci
    if si is None: si = lsi.si
    if opt is None: opt = lsi.opt
    if si.ndim==1:
        nroots_si=1
    else:
        assert (si.ndim==2)
        nroots_si = si.shape[1]
    si = si.reshape (-1,nroots_si)
    nelec_frs = lsi.get_nelec_frs ()
    h0, h1, h2 = lsi.ham_2q (mo_coeff=mo_coeff, soc=soc)
    hop = op[opt].gen_contract_op_si_hdiag (
        lsi, h1, h2, ci, nelec_frs, soc=soc
    )[0]
    e_tot = lib.einsum ('ip,ip->p', (hop (si) + h0*si), si.conj ())
    if nroots_si==1: e_tot=e_tot[0]
    return e_tot

class LASSI(lib.StreamObject):
    '''
    LASSI Method class
    '''
    def __init__(self, las, mo_coeff=None, ci=None, soc=False, break_symmetry=False, opt=1,
                 **kwargs):
        from mrh.my_pyscf.mcscf.lasci import LASCINoSymm
        if isinstance(las, LASCINoSymm): self._las = las
        else: raise RuntimeError("LASSI requires las instance")
        if mo_coeff is None: mo_coeff = las.mo_coeff
        if ci is None: ci = las.ci
        self.mo_coeff, self.ci = mo_coeff, ci
        # indiscriminate "dict update" from las is bad practice. not doing that anymore
        # Wave function configuration data from las parent
        self.mol = las.mol
        self.ncore, self.ncas = las.ncore, las.ncas
        self.nfrags, self.nroots = las.nfrags, las.nroots
        self.ncas_sub, self.nelecas_sub, self.fciboxes = las.ncas_sub, las.nelecas_sub, las.fciboxes
        self.nelecas = sum (self.nelecas_sub)
        self.weights, self.e_states, self.e_lexc = las.weights, las.e_states, las.e_lexc
        self.converged = las.converged
        # I/O data from las parent
        self.stdout, self.verbose, self.chkfile = las.stdout, las.verbose, las.chkfile
        # General config data from las parent
        self.max_memory = las.max_memory
        keys = set(('e_roots', 'si', 's2', 'nelec', 'wfnsym', 'rootsym', 'break_symmetry', 'soc', 'opt'))
        self.e_roots = None
        self.si = None
        self.s2 = None
        self.nelec = None
        self.wfnsym = None
        self.rootsym = None
        self.break_symmetry = break_symmetry
        self.soc = soc
        self.opt = opt
        self.level_shift_si = LEVEL_SHIFT_SI
        self.nroots_si = NROOTS_SI
        self.converged_si = False
        self._keys = set((self.__dict__.keys())).union(keys)

    def kernel(self, mo_coeff=None, ci=None, veff_c=None, h2eff_sub=None, orbsym=None, soc=None,\
               break_symmetry=None, opt=None,  **kwargs):
        if soc is None: soc = self.soc
        if break_symmetry is None: break_symmetry = self.break_symmetry
        if opt is None: opt = self.opt
        log = lib.logger.new_logger (self, self.verbose)
        t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
        if not self.converged:
            log.warn ('LASSI state preparation step not converged!')
        e_roots, si = lassi(self, mo_coeff=mo_coeff, ci=ci, veff_c=veff_c, h2eff_sub=h2eff_sub, orbsym=orbsym, \
                            soc=soc, break_symmetry=break_symmetry, opt=opt)
        self.e_roots = e_roots
        self.si, self.s2, self.nelec, self.wfnsym, self.rootsym, self.break_symmetry, self.soc  = \
            si, si.s2, si.nelec, si.wfnsym, si.rootsym, si.break_symmetry, si.soc
        log.timer ('LASSI matrix-diagonalization kernel', *t0)
        return self.e_roots, self.si

    def ham_2q (self, mo_coeff=None, veff_c=None, h2eff_sub=None, soc=0):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        return ham_2q (self, mo_coeff, veff_c=veff_c, h2eff_sub=h2eff_sub, soc=soc)

    def get_nelec_frs (self, las=None):
        if las is None: las = self
        from mrh.my_pyscf.mcscf.lasci import get_nelec_frs
        return get_nelec_frs (las)

    def get_smult_fr (self, las=None):
        if las is None: las = self
        from mrh.my_pyscf.mcscf.lasci import get_space_info
        return get_space_info (las)[2].T

    def get_sym_fr (self, las=None):
        if las is None: las = self
        from mrh.my_pyscf.mcscf.lasci import get_sym_fr
        return get_sym_fr (las)

    def get_lroots (self, ci=None):
        if ci is None: ci = self.ci
        return get_lroots (ci)

    def get_nprods (self, ci=None):
        lroots = self.get_lroots (ci=ci)
        return np.sum (np.prod (lroots, axis=0))

    def get_sivec_fermion_spin_shuffle (self, si=None, ci=None):
        from mrh.my_pyscf.lassi.sitools import sivec_fermion_spin_shuffle
        if si is None: si = self.si
        lroots = self.get_lroots (ci=ci)
        nelec_frs = self.get_nelec_frs ()
        return sivec_fermion_spin_shuffle (si, nelec_frs, lroots)

    def get_sivec_vacuum_shuffle (self, state=None, nelec_vac=None, si=None, ci=None):
        '''Define a particular number of electrons in each fragment as the vacuum,
        set the signs of the LAS basis functions accordingly and in fragment-major
        order, and return the correspondingly-modified SI vector.
    
        Kwargs:
            state: integer
                Index of the rootspace identified as the new vacuum. Required if
                nelec_vac is unset.
            nelec_vac: ndarray of shape (nfrags)
                Number of electrons (spinless) in each fragment in the new vacuum.
                Defaults to self.get_nelec_frs ()[:,state,:].sum (1)
            si: ndarray of shape (nstates,*)
                SI vectors; taken from self if omitted
            ci: list of list of ndarrays
                CI vectors; taken from self if omitted
 
        Returns:
            si1: ndarray of shape (nstates,*)
                si0 with permuted row signs corresponding to nelec_vac electrons in
                each fragment in the vacuum and the fermion creation operators in
                fragment-major order
        '''
        from mrh.my_pyscf.lassi.sitools import sivec_vacuum_shuffle
        if si is None: si = self.si
        lroots = self.get_lroots (ci=ci)
        nelec_frs = self.get_nelec_frs ()
        return sivec_vacuum_shuffle (si, nelec_frs, lroots, nelec_vac=nelec_vac, state=state)

    def make_casdm12s (self, ci=None, si=None, state=None, weights=None, spaces=None, opt=None):
        if ci is None: ci = self.ci
        if si is None: si = self.si
        if opt is None: opt = self.opt
        nstates = 1 if si.ndim==1 else si.shape[1]
        if nstates==1:
            if state is None: state=0
            if si.ndim==1: si=si[:,None]
        if state is None:
            dm1s, dm2s = roots_make_rdm12s (self, ci, si, spaces=spaces, opt=opt)
            if weights is not None:
                dm1s = lib.einsum ('r,rspq->spq', weights, dm1s)
                dm2s = lib.einsum ('r,rspqtxy->spqtxy', weights, dm2s)
            return dm1s, dm2s
        else:
            return root_make_rdm12s (self, ci, si, state=state, spaces=spaces, opt=opt)

    def make_casdm12 (self, ci=None, si=None, state=None, weights=None, spaces=None, opt=None):
        dm1s, dm2s = self.make_casdm12s (ci=ci, si=si, state=state, weights=weights, spaces=spaces,
                                         opt=opt)
        return dm1s.sum (0), dm2s.sum ((0,3))

    def make_casdm2 (self, ci=None, si=None, state=None, weights=None, spaces=None, opt=None):
        return self.make_casdm12 (ci=ci, si=si, state=state, weights=weights, spaces=spaces,
                                  opt=opt)[1]

    def trans_casdm12s (self, ci=None, si_bra=None, si_ket=None, state=None, weights=None,
                        spaces=None, opt=None):
        if ci is None: ci = self.ci
        if si_bra is None: si_bra = self.si
        if si_ket is None: si_ket = self.si
        if opt is None: opt = self.opt
        nstates = 1 if si_bra.ndim==1 else si_bra.shape[1]
        nstates_ket = 1 if si_ket.ndim==1 else si_ket.shape[1]
        assert (nstates==nstates_ket)
        if nstates==1:
            if state is None: state=0
            if si_bra.ndim==1: si_bra=si_bra[:,None]
            if si_ket.ndim==1: si_ket=si_ket[:,None]
        if state is None:
            dm1s, dm2s = roots_trans_rdm12s (self, ci, si_bra, si_ket, spaces=spaces, opt=opt)
            if weights is not None:
                dm1s = lib.einsum ('r,rspq->spq', weights, dm1s)
                dm2s = lib.einsum ('r,rspqtxy->spqtxy', weights, dm2s)
            return dm1s, dm2s
        else:
            return root_trans_rdm12s (self, ci, si_bra, si_ket, state=state, spaces=spaces,
                                      opt=opt)

    def trans_casdm12 (self, ci=None, si_bra=None, si_ket=None, state=None, weights=None,
                       spaces=None, opt=None):
        dm1s, dm2s = self.trans_casdm12s (ci=ci, si_bra=si_bra, si_ket=si_ket, state=state,
                                          weights=weights, spaces=spaces, opt=opt)
        return dm1s.sum (0), dm2s.sum ((0,3))

    def make_rdm1s (self, mo_coeff=None, ci=None, si=None, state=None, weights=None, spaces=None,
                    opt=None):
        if mo_coeff is None: mo_coeff=self.mo_coeff
        casdm1s = self.make_casdm12s (ci=ci, si=si, state=state, weights=weights, spaces=spaces,
                                      opt=opt)[0]
        mo_core = mo_coeff[:,:self.ncore]
        mo_cas = mo_coeff[:,self.ncore:][:,:self.ncas]
        dm_core = mo_core @ mo_core.conj ().T
        dm1s = lib.einsum ('up,vq,spq->suv', mo_cas, mo_cas.conj (), casdm1s) + dm_core[None,:,:]
        return dm1s

    def make_rdm1 (self, mo_coeff=None, ci=None, si=None, state=None, weights=None, spaces=None,
                   opt=None):
        dm1s = self.make_rdm1s (mo_coeff=mo_coeff, ci=ci, si=si, state=state, weights=weights,
                                spaces=spaces, opt=opt)
        return dm1s[0] + dm1s[1]

    def analyze (self, state=0, **kwargs):
        from mrh.my_pyscf.lassi.sitools import analyze
        return analyze (self, self.si, state=state, **kwargs)

    def reset (self, mol=None):
        if mol is not None:
            self.mol = mol
        self._las.reset (mol)

    dump_chk = chkfile.dump_lsi
    load_chk = load_chk_ = chkfile.load_lsi_

    def get_init_guess_si (self, hdiag, nroots, si1):
        return get_init_guess_si (hdiag, nroots, si1)

    energy_tot = energy_tot

    def get_raw2orth (self, ci=None, nelec_frs=None, soc=None, opt=None, _get_ovlp=None):
        if ci is None: ci = self.ci
        if soc is None: soc = self.soc
        if opt is None: opt = self.opt
        if nelec_frs is None: nelec_frs = self.get_nelec_frs ()
        n = self.ncas
        h1 = np.zeros ([n,n])
        h2 = np.zeros ([n,n,n,n])
        if _get_ovlp is None:
            _get_ovlp = op[opt].gen_contract_op_si_hdiag (
                self, h1, h2, ci, nelec_frs, soc=soc
            )[4]
        return citools.get_orth_basis (ci, self.ncas_sub, nelec_frs, _get_ovlp=_get_ovlp)

    def get_casscf_eris (self, mo_coeff=None):
        if mo_coeff is None: mo_coeff=self.mo_coeff
        las = self._las
        from mrh.my_pyscf.mcscf import _DFLASCI
        from pyscf.mcscf import mc_ao2mo, df
        if isinstance (las, _DFLASCI):
            eris = df._ERIS (las, mo_coeff, las.with_df)
        else:
            eris = mc_ao2mo._ERIS (las, mo_coeff, method='incore', level=2)
        return eris

