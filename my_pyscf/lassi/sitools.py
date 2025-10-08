import numpy as np
from pyscf import lib, symm
from scipy import linalg
from mrh.my_pyscf.mcscf.lasci import get_space_info
from mrh.my_pyscf.lassi.citools import get_lroots, get_rootaddr_fragaddr, umat_dot_1frag_, _umat_dot_1frag
from mrh.my_pyscf.lassi.lassi import root_make_rdm12s, LASSI, ham_2q
from mrh.my_pyscf.lassi.op_o1.utilities import fermion_spin_shuffle

def decompose_sivec_by_rootspace (las, si, ci=None):
    '''Decompose a set of LASSI vectors as

    si[i,:] = +sqrt(space_weights[P,:])*state_coeffs[P][a,:]

    Where "i" indexes the "a"th state in rootspace "P"'''
    if ci is None: ci=las.ci
    if si.ndim==1: si = si[:,None]
    lroots = get_lroots (ci)
    nstates = np.prod (lroots, axis=0)
    jj = np.cumsum (nstates)
    ii = jj - nstates
    nspaces = las.nroots
    nbas, nroots = si.shape
    space_coeffs = np.empty ((nspaces, nroots), dtype=si.dtype)
    state_coeffs = []
    ham_space = []
    idx_space = np.zeros((nspaces, nbas), dtype=bool)
    for space, (i, j) in enumerate (zip (ii, jj)):
        space_coeffs[space] = linalg.norm (si[i:j,:], axis=0)
        idx_space[space,i:j] = True
        idx = space_coeffs[space]>0
        state_coeffs.append (si[i:j,:].copy ())
        state_coeffs[-1][:,idx] /= space_coeffs[space][idx]
    return space_coeffs**2, state_coeffs, idx_space

def _make_sdm1 (sivec, lroots, site):
    '''Compute the 1-site reduced density matrix(es) for (a) wave function(s) of type

    |Psi> = sum_n sivec[n] |n0n1n2n3....>

    where nK < lroots[K] are nonnegative integes

    Args:
        sivec: ndarray of shape (np.prod (lroots), nroots)
            coefficients of the wave function(s) with site quantum numbers
            increasing from |00000...> in column-major order
        lroots: ndarray of shape (nsites)
            number of states on each site in the product-state basis
        site: integer
            site index for which to compute the density matrix

    Returns:
        sdm1: ndarray of shape (nroots,lroots[site],lroots[site])
            One-site reduced density matrix
    '''
    nsites = len (lroots)
    nroots, err = divmod (sivec.size, np.prod (lroots))
    if err: raise ValueError ("sivec.size % prod(lroots) = {}".format (err))
    sivec = np.asfortranarray (sivec)
    sivec = sivec.reshape (list(lroots)+[nroots,], order='F')
    idx = [site,nsites]
    if site<nsites-1: idx = list(range(site+1,nsites)) + idx
    if site>0: idx = list(range(site)) + idx
    sivec = sivec.transpose (*idx).reshape (-1, lroots[site],nroots)
    return lib.einsum ('api,aqi->ipq', sivec.conj(), sivec)

def _trans_sdm1 (sivec_bra, lroots_bra, sivec_ket, lroots_ket, ovlp, site):
    nsites = len (lroots_bra)
    nroots_bra, err = divmod (sivec_bra.size, np.prod (lroots_bra))
    if err: raise ValueError ("sivec_bra.size % prod(lroots_bra) = {}".format (err))
    nroots_ket, err = divmod (sivec_ket.size, np.prod (lroots_ket))
    if err: raise ValueError ("sivec_ket.size % prod(lroots_ket) = {}".format (err))
    assert (nroots_bra==nroots_ket)
    nroots = nroots_bra
    for isite, s0 in enumerate (ovlp):
        if isite==site: continue
        # TODO: the function below has to be modified for a non-square umat
        # s0 might have to be transposed
        sivec_ket = _umat_dot_1frag (sivec_ket, s0.T, lroots_ket, isite)
        lroots_ket[isite] = lroots_bra[isite]
    sivec_bra = np.asfortranarray (sivec_bra)
    sivec_bra = sivec_bra.reshape (list(lroots_bra)+[nroots,], order='F')
    sivec_bra = np.moveaxis (sivec_bra, site, -2)
    sivec_bra = sivec_bra.reshape (-1,lroots_bra[site],nroots)
    sivec_ket = np.asfortranarray (sivec_ket)
    sivec_ket = sivec_ket.reshape (list(lroots_ket)+[nroots,], order='F')
    sivec_ket = np.moveaxis (sivec_ket, site, -2)
    sivec_ket = sivec_ket.reshape (-1,lroots_ket[site],nroots)
    return lib.einsum ('api,aqi->ipq', sivec_bra.conj(), sivec_ket)

def _make_sdm2 (sivec, lroots, site1, site2):
    '''Compute the 2-site reduced density matrix(es) for (a) wave function(s) of type

    |Psi> = sum_n sivec[n] |n0n1n2n3....>

    where nK < lroots[K] are nonnegative integes

    Args:
        sivec: ndarray of shape (np.prod (lroots), nroots)
            coefficients of the wave function(s) with site quantum numbers
            increasing from |00000...> in column-major order
        lroots: ndarray of shape (nsites)
            number of states on each site in the product-state basis
        site1: integer
            first site index for which to compute the density matrix
        site2: integer
            second site index for which to compute the density matrix

    Returns:
        sdm2: ndarray of shape (nroots,lroots[site1],lroots[site2],lroots[site1],lroots[site2])
            Two-site reduced density matrix
    '''
    nsites = len (lroots)
    nroots, err = divmod (sivec.size, np.prod (lroots))
    if err: raise ValueError ("sivec.size % prod(lroots) = {}".format (err))
    sivec = np.asfortranarray (sivec)
    sivec = sivec.reshape (list(lroots)+[nroots,], order='F')
    sivec = np.moveaxis (sivec, (site1,site2), (-3,-2))
    sivec = sivec.reshape (-1,lroots[site1],lroots[site2],nroots)
    return lib.einsum ('apqi,arsi->iprqs', sivec.conj(), sivec)


def make_sdm1 (lsi, iroot, ifrag, ci=None, si=None):
    if ci is None: ci = lsi.ci
    if si is None: si = lsi.si
    lroots = get_lroots (ci).T[iroot]
    space_weights, state_coeffs, idx_space = decompose_sivec_by_rootspace (
        lsi, si
    )
    w = space_weights[iroot]
    state_coeffs = state_coeffs[iroot]
    return _make_sdm1 (state_coeffs, lroots, ifrag) * w[:,None,None]
   
def get_rootspace_central_moment (las, space_weights, n=1):
    '''Compute either the mean (if n==1) or the nth central moment
    of the quantum numbers that define the rootspaces of a LASSI
    wave function. This means the average local charge and the two
    average local spin quantum numbers (average point group is not
    defined)

    Args:
        las: instance of :class:`LASCINoSymm`
        space_weights: ndarray of shape (las.nroots,nvec)
            Rootspace coefficients from a LASSI wave function
            (see `decompose_sivec_by_rootspace`)

    Kwargs:
        n: integer
            Moment to return; if n==1, returns the mean

    Returns:
        charges: ndarray of shape (nvec,las.nfrags)
            nth moment of the local charges
        spins: ndarray of shape (nvec,las.nfrags)
            nth moment of the local spin polarizations
        smults: ndarray of shape (nvec,las.nfrags)
            nth moment of the local spin magnitudes'''
    if space_weights.ndim<2:
        space_weights = space_weights[:,None]
    nvec = space_weights.shape[1]
    charges, spins, smults, wfnsyms = get_space_info (las)
    c1 = np.dot (charges.T, space_weights).T
    m1 = np.dot (spins.T, space_weights).T
    s1 = np.dot (smults.T, space_weights).T
    if n==1:
        return c1, m1, s1
    c2 = lib.einsum ('frv,rv->vf', np.power (charges.T[:,:,None]-c1.T[:,None,:], n), space_weights)
    m2 = lib.einsum ('frv,rv->vf', np.power (spins.T[:,:,None]-m1.T[:,None,:], n), space_weights)
    s2 = lib.einsum ('frv,rv->vf', np.power (smults.T[:,:,None]-s1.T[:,None,:], n), space_weights)
    return c2, m2, s2


def _print_states (log, iroot, space_weights, state_coeffs, lroots, print_all_but=1e-8):
    nstates = state_coeffs.shape[1]
    space_coeffs = np.sqrt (space_weights)
    nfrags = len (lroots)
    nprods = np.prod (lroots)
    state_weights = (state_coeffs*state_coeffs).sum (1)/nstates
    running_weight = 1
    idx = np.argsort (-state_weights)
    addrs = np.stack (np.meshgrid (*[np.arange(l) for l in lroots[::-1]],
                                  indexing='ij'), axis=0)
    addrs = addrs.reshape (nfrags, nprods)[::-1,:].T
    addrs_len = np.ceil (np.log10 (lroots.astype (float))).astype (int)
    addrs_len = np.maximum (1, addrs_len)
    lbl_len = sum (addrs_len)
    if np.all (addrs_len==1):
        fmt_str0 = ''.join (['{:1d}',]*nfrags)
    else:
        lbl_len += (nfrags-1) # periods between fragment indices
        fmt_str0 = '.'.join (['{:d}',]*nfrags)
    lbl_len = max (3,lbl_len)
    fmt_str1 = ' {:>' + str (lbl_len) + 's}: ' + ' '.join (['{:10.3e}',]*nstates)
    log.info (fmt_str1.format ('Fac', *space_coeffs))
    for ix, iprod in enumerate (idx):
        lbl_str = fmt_str0.format (*addrs[iprod])
        log.info (fmt_str1.format (lbl_str, *state_coeffs[iprod]))
        running_weight -= state_weights[iprod]
        if running_weight < print_all_but: break
    if ix+1<nprods:
        log.info ("Remaining %d ENVs in rootspace %d have combined average weight = %e",
                  nprods-ix-1, iroot, running_weight)
    else:
        log.info ("All %d ENVs in rootspace %d accounted for", nprods, iroot)
    return

def analyze (las, si, ci=None, state=0, print_all_but=1e-8, lbasis='primitive', ncsf=10,
             return_metrics=False, do_natorb=True):
    '''Print out analysis of LASSI result in terms of average quantum numbers
    and density matrix analyses of the lroots in each rootspace

    Args:
        las: instance of :class:`LASCINoSymm`
        si: ndarray of shape (nstates,nstates)

    Kwargs:
        ci: list of list of ndarray
            Fragment CI vectors. Taken from las if not provided
        state: integer or index array
            indicates which columns of si to analyze. Indicated columns are
            averaged together for the lroots analysis
        print_all_but: float
            continue density-matrix analysis printouts until all
            but this weight of the wave function(s) is accounted for. Set
            to less than zero to print everything.
        lbasis : 'primitive' or 'Schmidt'
            Basis in which local fragment CI vectors are represented.
            The primitive basis is the CI vectors literally stored on ci.
            The Schmidt basis diagonalizes the one-site reduced density matrix.
        ncsf : integer
            Number of leading CSFs of each basis function to print.
        return_metrics: logical
            If True, returns space_weights, navg, maxw, and entr arrays; see below.
        do_natorb: logical
            If True, include a natural-orbital analysis (which is sometimes very
            slow due to poor optimization of LASSI op_o1 RDM build functions)

    Returns:
        ci1: list of list of ndarray
            Fragment CI vectors from the analysis. If lbasis='Schmidt', they are
            rotated into the Schmidt basis
        si1: ndarray of shape (ndim, len (state))
            SI vectors. If lbasis='Schmidt', they are rotated into the Schmidt basis
        space_weights: ndarray of shape (nroots); optional
            Average weight in each rootspace
        navg: ndarray of shape (nroots, nfrags); optional
            Expectation value of the principal quantum number of each fragment in each rootspace
            If print_all_but >= 0, some rows may be omitted (set to -1)
        maxw: ndarray of shape (nroots, nfrags); optional
            Value of the maximum weight on any one eigenstate for each fragment in each rootspace
            If print_all_but >= 0, some rows may be omitted (set to -1)
        entr: ndarray of shape (nroots, nfrags); optional
            Von Neumann entropy of each fragment, considering each rootspace separately
            If print_all_but >= 0, some rows may be omitted (set to -1)
    '''
    analyze_moments (las, si, ci=ci, do_natorb=do_natorb)
    if 'prim' in lbasis.lower (): lbasis = 'primitive'
    elif 'schmidt' in lbasis.lower (): lbasis = 'Schmidt'
    else:
        raise RuntimeError (
            f"lbasis = {lbasis} undefined (supported values: 'primitive' or 'Schmidt')"
        )
    if ci is None: ci = las.ci
    ci0 = ci
    states = np.atleast_1d (state)
    nstates = len (states)
    log = lib.logger.new_logger (las, las.verbose)
    space_weights, state_coeffs, idx_space = decompose_sivec_by_rootspace (
        las, si
    )
    avg_weights = space_weights[:,states].sum (1) / nstates
    c, m, s, w = get_space_info (las)
    nelelas = np.array ([sum (n) for n in las.nelecas_sub])
    lroots = get_lroots (ci0).T
    running_weight = 1
    fmt_str = " {:4s}  {:>7s}  {:>4s}  {:>3s}  {:>6s}  {:11s}  {:>8s}"
    header = fmt_str.format ("Frag", "Nelec", "2S+1", "Ir", "<n>", "Max(weight)", "Entropy")
    fmt_str = " {:4d}  {:>7s}  {:>4d}  {:>3s}  {:6.3f}  {:>11.4f}  {:8f}"
    ci1 = [[ci0[ifrag][iroot].view () for iroot in range (las.nroots)] for ifrag in range (las.nfrags)]
    si1 = si.copy ()
    navg = -np.ones ((las.nroots, las.nfrags), dtype=float)
    maxw = -np.ones ((las.nroots, las.nfrags), dtype=float)
    entr = -np.ones ((las.nroots, las.nfrags), dtype=float)
    for ix, iroot in enumerate (np.argsort (-avg_weights)):
        log.info ("Rootspace %d with averaged weight %9.3e", iroot, avg_weights[iroot])
        log.info (header)
        addr_shape = list (lroots[iroot][::-1]) + [state_coeffs[iroot].shape[-1]]
        flat_shape = state_coeffs[iroot].shape
        coeffs = state_coeffs[iroot].copy ().reshape (*addr_shape)
        ci_f = []
        for ifrag in range (las.nfrags):
            sdm = _make_sdm1 (state_coeffs[iroot][:,states], lroots[iroot], ifrag).sum (0) / nstates
            dens = sdm.diagonal ()
            navg[iroot,ifrag] = np.dot (np.arange (len (dens)), dens)
            maxw[iroot,ifrag] = np.amax (dens)
            evals, evecs = linalg.eigh (-sdm)
            evals = -evals
            evals = evals[evals>0]
            entr[iroot,ifrag] = abs(np.dot (evals, np.log (evals)))
            nelec = "{}a+{}b".format ((nelelas[ifrag]-c[iroot,ifrag]+m[iroot,ifrag])//2,
                                      (nelelas[ifrag]-c[iroot,ifrag]-m[iroot,ifrag])//2)
            ir = symm.irrep_id2name (las.mol.groupname, w[iroot][ifrag])
            ci = np.asarray (ci0[ifrag][iroot])
            if ci.ndim==2: ci=ci[None,:,:]
            if 'schmidt' in lbasis.lower ():
                coeffs = np.tensordot (coeffs, evecs, axes=((las.nfrags-1-ifrag,),(0,)))
                axes_order = list (range (las.nfrags))
                axes_order.insert (las.nfrags-1-ifrag, las.nfrags)
                coeffs = coeffs.transpose (*axes_order)
                ci = np.tensordot (evecs.T, ci, axes=1)
            ci_f.append (ci)
            ci1[ifrag][iroot] = ci
            log.info (fmt_str.format (ifrag, nelec, s[iroot][ifrag], ir, navg[iroot,ifrag],
                      maxw[iroot,ifrag], entr[iroot,ifrag]))
        coeffs = coeffs.reshape (flat_shape)
        si1[idx_space[iroot],:] = np.sqrt (space_weights[iroot]) * coeffs[:,:]
        log.info ("Wave function(s) in rootspace %d in local %s basis:", iroot, lbasis)
        _print_states (log, iroot, space_weights[iroot,states], coeffs[:,states], lroots[iroot],
                       print_all_but=print_all_but)
        if ncsf:
            for ifrag, ci in enumerate (ci_f):
                log.info (("Leading %d CSFs of %d local %s basis functions for fragment "
                           "%d in rootspace %d:"), ncsf, len (ci), lbasis, ifrag, iroot)
                analyze_basis (las, ci=ci, space=iroot, frag=ifrag, npr=ncsf)
        running_weight -= avg_weights[iroot]
        if running_weight < print_all_but: break

    if ix+1<las.nroots:
        log.info ("Remaining %d rootspaces have combined weight = %e",
                  las.nroots-ix-1, running_weight)
        if return_metrics:
            log.warn (("Not all rootspaces examined because their weights are too small. Metrics "
                       "for omitted rootspaces are set to -1. Set print_all_but=-1 to ensure all "
                       "rootspaces are examined."))
    else:
        log.info ("All %d rootspaces accounted for", las.nroots)

    average_metrics_over_spin_polarization (las, avg_weights, navg, maxw, entr)

    if return_metrics:
        space_weights = avg_weights
        return ci1, si1, space_weights, navg, maxw, entr
    else:
        return ci1, si1

def analyze_moments (las, si, ci=None, state=0, do_natorb=True):
    '''The part of analyze that would be in common between LASSI and LASSIS'''
    if ci is None: ci = las.ci
    ci0 = ci
    states = np.atleast_1d (state)
    nstates = len (states)

    log = lib.logger.new_logger (las, las.verbose)
    if do_natorb:
        log.info ("Natural-orbital analysis for state(s) %s", str (state))
        casdm1s = root_make_rdm12s (las, ci, si, state=state)[0]
        if nstates > 1:
            casdm1s = casdm1s.sum (0) / nstates
        casdm1 = casdm1s.sum (0)
        if isinstance (las, LASSI):
            no_coeff, no_ene, no_occ = las._las.canonicalize (natorb_casdm1=casdm1)[:3]
        else:
            no_coeff, no_ene, no_occ = las.canonicalize (natorb_casdm1=casdm1)[:3]

    log.info ("Analyzing LASSI vectors for states = %s",str(state))
    for st in states:
        print_wfn_leading (las, state=st, ci=ci0, si=si) 

    log.info ("Average quantum numbers:")
    space_weights, state_coeffs, idx_space = decompose_sivec_by_rootspace (
        las, si
    )
    nelelas = np.array ([sum (n) for n in las.nelecas_sub])
    c, m, smults = get_rootspace_central_moment (las, space_weights[:,states])
    neleca = .5*(nelelas[None,:]-c+m)
    nelecb = .5*(nelelas[None,:]-c-m)
    for na, nb, s, st in zip (neleca, nelecb, smults, states):
        log.info ("State %d:", st)
        log.info ("Neleca = %s", str (na))
        log.info ("Nelecb = %s", str (nb))
        log.info ("Smult = %s", str (s))

    def log_qn_spread (qn_table, weights):
        qns = np.unique (qn_table)
        fmt_str = '{:4d} ' + ' '.join (['{:10.3e}',]*las.nfrags)
        for qn in qns:
            row = [np.sum (weights[qn_table[:,ifrag]==qn]) for ifrag in range (las.nfrags)]
            log.info (fmt_str.format (qn, *row))

    c, m, s, w = get_space_info (las)
    avg_weights = space_weights[:,states].sum (1) / nstates
    log.info ("Spread of charges:")
    log_qn_spread (c, avg_weights)
    log.info ("Spread of spin multiplicities:")
    log_qn_spread (s, avg_weights)

def average_metrics_over_spin_polarization (las, weights, navg, maxw, entr):
    log = lib.logger.new_logger (las, las.verbose)
    c, m, s, w = get_space_info (las)
    def log_metric (data):
        fmt_str = '{:4s} {:4s} ' + ' '.join (['Frag {:<5d}',]*las.nfrags)
        log.info (fmt_str.format ('Chrg', 'Smlt', *list(range(las.nfrags))))
        idx_omitted = np.any (data==-1,axis=1)
        for tc in np.unique (c):
            for ts in np.unique (s):
                line = '{:4d} {:4d}'.format (tc, ts)
                row = []
                doline = False
                for ifrag in range (las.nfrags):
                    idx = c[:,ifrag]==tc
                    idx &= s[:,ifrag]==ts
                    idx &= ~idx_omitted
                    if np.any (idx):
                        d = data[:,ifrag][idx]
                        w = weights[idx]
                        line += ' {:10.3e}'.format (np.average (d, weights=w))
                        doline = True
                    else:
                        line += ' {:>10s}'.format ('-')
                if doline: log.info (line)
    for lbl, metric in zip (('<n>', 'max(weight)', 'entropy'), (navg, maxw, entr)):
        log.info ('Average over spin polarization of %s', lbl)
        log_metric (metric)

def print_wfn_leading (lsi, state=0, ci=None, si=None, ncoeffs=20):
    if si is None: si = lsi.si
    if ci is None: ci = lsi.ci
    si = si[:,state]
    log = lib.logger.new_logger (lsi, lsi.verbose)
    rootaddr, envaddr = get_rootaddr_fragaddr (get_lroots (ci))
    idx = np.argsort (-si*si)
    idx = idx[:ncoeffs]
    si, rootaddr, envaddr = si[idx], rootaddr[idx], envaddr[:,idx].T
    log.info ("Printing leading %d SI addresses and coeffs of state %d (wgt = %9.3e)",
              ncoeffs, state, np.dot (si, si))
    if np.amax (envaddr) < 10:
        env_fmt = ''.join (['{:1d}',]*lsi.nfrags)
    else: 
        env_fmt = '.'.join (['{:1d}',]*lsi.nfrags)
    envaddr = [env_fmt.format (*env) for env in envaddr]
    for coeff, root, env in zip (si, rootaddr, envaddr):
        log.info ("%5d|%s : %10.3e", root, env, coeff)

def analyze_basis (las, ci=None, space=0, frag=0, npr=10):
    '''Print out the many-electron wave function(s) in terms of CSFs for a specific
    fragment in a specific LAS rootspace'''
    if ci is None: ci = las.ci
    if isinstance (ci, list): ci = ci[frag][space]
    norb = las.ncas_sub[frag]
    fcibox = las.fciboxes[frag]
    fcisolver = fcibox.fcisolvers[space]
    transformer = fcisolver.transformer
    nelec = fcibox._get_nelec (fcisolver, las.nelecas_sub[frag])
    log = lib.logger.new_logger (las, las.verbose)
    l, c = transformer.printable_largest_csf (ci, npr, isdet=True)
    nstates = len (l)
    colwidth = 12 + norb
    headlen = 7 + 2*int(np.ceil(np.log10(nstates)))
    colwidth = max (colwidth, headlen)
    ncols = 80 // (colwidth+3)
    ntabs = (nstates + (ncols-1)) // ncols
    fmt0 = '{:' + str(colwidth) + 's}'
    fmt1_head = 'State {:d}/' + str (nstates)
    fmt1_body = '{:' + str (norb) + 's} : {:9.2e}'
    npr = min (npr, transformer.ncsf)
    for itab in range (ntabs):
        state_range = range (itab*ncols,min(nstates,(itab+1)*ncols))
        line = ''
        for state in state_range:
            entry = fmt1_head.format (state)
            line += fmt0.format (entry) + '   '
        log.info (line)
        for ipr in range (npr):
            line = ''
            for state in state_range:
                entry = fmt1_body.format (l[state][ipr], c[state][ipr])
                line += fmt0.format (entry) + '   '
            log.info (line)


def analyze_ham (las, si, e_roots, ci=None, state=0, soc=0, print_all_but=1e-8):
    '''Print out analysis of LASSI result in terms of Hamiltonian matrix
    elements

    Args:
        las: instance of :class:`LASCINoSymm`
        si: ndarray of shape (nstates,nstates)
            SI vectors
        e_roots: sequence of length (nstates,)
            energies
        soc: integer
            Order of spin-orbit coupling included in the Hamiltonian. Currently
            only 0 or 1 is supported.

    Kwargs:
        ci: list of list of ndarray
            Fragment CI vectors. Taken from las if not provided
        state: integer or index array
            indicates which columns of si to analyze. Indicated columns are
            averaged together for the lroots analysis
        print_all_but: float
            continue density-matrix analysis printouts until all
            but this weight of the wave function(s) is accounted for. Set
            to zero to print everything.
    '''
    if ci is None: ci = las.ci
    ci0 = ci
    h0 = ham_2q (las, las.mo_coeff, soc=soc)[0]
    e_roots = np.asarray (e_roots) - h0

    log = lib.logger.new_logger (las, las.verbose)
    log.info ("Analyzing LASSI Hamiltonian for states = %s",str(state))

    space_weights, state_coeffs, idx_space = decompose_sivec_by_rootspace (
        las, si
    )
    states = np.atleast_1d (state)
    nelelas = np.array ([sum (n) for n in las.nelecas_sub])

    log.info ("Continue until 1-%e of wave function(s) accounted for", print_all_but)
    nstates = len (states)
    avg_weights = space_weights[:,states].sum (1) / nstates
    lroots = get_lroots (ci0).T
    c, m, s, w = get_space_info (las)
    fmt_str = " {:4s}  {:>7s}  {:>4s}  {:>3s}  {:>6s}  {:11s}  {:>8s}"
    header = fmt_str.format ("Frag", "Nelec", "2S+1", "Ir", "<n>", "Max(weight)", "Entropy")
    fmt_str = " {:4d}  {:>7s}  {:>4d}  {:>3s}  {:6.3f}  {:>11.4f}  {:8f}"
    log.info ("E(core) = %16.9e", h0)
    h = (si * e_roots[None,:]) @ si.conj ().T
    for iroot in range (las.nroots):
        log.info ("Rootspace %d with averaged weight %9.3e", iroot, avg_weights[iroot])
        log.info (header)
        idx_p = idx_space[iroot]
        idx_q = ~idx_p
        h_pp = h[idx_p,:][:,idx_p]
        h_pq = h[idx_p,:][:,idx_q]
        h_qq = h[idx_q,:][:,idx_q]
        hdiag0 = np.diag (h_pp)
        hdiag_i = np.zeros ((h_pp.shape[0],len(states)))
        for p, h_q in enumerate (h_pq):
            for i, e_i in enumerate (e_roots[states]):
                A = e_i*np.eye (h_qq.shape[0]) - h_qq
                hdiag_i[p,i] = np.dot (h_q.conj (), linalg.solve (A, h_q))
        h_pp = h_pq = h_qq = None 

        for ifrag in range (las.nfrags):
            sdm = _make_sdm1 (state_coeffs[iroot][:,states], lroots[iroot], ifrag).sum (0) / nstates
            dens = sdm.diagonal ()
            navg = np.dot (np.arange (len (dens)), dens)
            maxw = np.amax (dens)
            evals, evecs = linalg.eigh (-sdm)
            evals = -evals
            evals = evals[evals>0]
            entr = abs(np.dot (evals, np.log (evals)))
            nelec = "{}a+{}b".format ((nelelas[ifrag]-c[iroot,ifrag]+m[iroot,ifrag])//2,
                                      (nelelas[ifrag]-c[iroot,ifrag]-m[iroot,ifrag])//2)
            ir = symm.irrep_id2name (las.mol.groupname, w[iroot][ifrag])
            log.info (fmt_str.format (ifrag, nelec, s[iroot][ifrag], ir, navg, maxw, entr))
        log.info ("Diagonal Hamiltonian elements in rootspace %d:", iroot)
        _print_hdiag (log, iroot, state_coeffs[iroot][:,states], lroots[iroot],
                      hdiag0, hdiag_i, states, print_all_but=print_all_but)

    return

def _print_hdiag (log, iroot, state_coeffs, lroots, hdiag0, hdiag_i,
                  states, print_all_but=1e-8):
    e_ref = np.mean (hdiag0) # insensitive to changes in basis -> easier to read
    log.info ('E(ref) - E(core) = %16.9e', e_ref)
    hdiag0 = hdiag0 - e_ref
    hdiag_i = hdiag0[:,None] + hdiag_i
    nstates = state_coeffs.shape[1]
    nfrags = len (lroots)
    nprods = np.prod (lroots)
    state_weights = (state_coeffs*state_coeffs).sum (1)/nstates
    running_weight = 1
    idx = np.argsort (-state_weights)
    addrs = np.stack (np.meshgrid (*[np.arange(l) for l in lroots[::-1]],
                                  indexing='ij'), axis=0)
    addrs = addrs.reshape (nfrags, nprods)[::-1,:].T
    addrs_len = np.ceil (np.log10 (lroots.astype (float))).astype (int)
    addrs_len = np.maximum (1, addrs_len)
    lbl_len = sum (addrs_len)
    if np.all (addrs_len==1):
        fmt_str0 = ''.join (['{:1d}',]*nfrags)
    else:
        lbl_len += (nfrags-1) # periods between fragment indices
        fmt_str0 = '.'.join (['{:d}',]*nfrags)
    lbl_len = max (3,lbl_len)
    fmt_str1 = ' {:>' + str (lbl_len+1) + 's} ' + ' '.join (['{:>10s}',]*(nstates+1))
    headers = ['heff[{:d}]'.format (state) for state in states]
    log.info (fmt_str1.format ('Id', 'ham_pp', *headers))
    fmt_str1 = ' {:>' + str (lbl_len) + 's}: ' + ' '.join (['{:10.3e}',]*(nstates+1))
    for ix, iprod in enumerate (idx):
        lbl_str = fmt_str0.format (*addrs[iprod])
        log.info (fmt_str1.format (lbl_str, hdiag0[iprod], *list(hdiag_i[iprod,:])))
        running_weight -= state_weights[iprod]
        if running_weight < print_all_but: break
    if ix+1<nprods:
        log.info ("Remaining %d ENVs in rootspace %d have combined average weight = %e",
                  nprods-ix-1, iroot, running_weight)
    else:
        log.info ("All %d ENVs in rootspace %d accounted for", nprods, iroot)
    return

def sivec_fermion_spin_shuffle (si0, nelec_frs, lroots):
    '''Permute the signs of rows of the SI vector to account for the difference in convention
    between

    ... a2' a1' a0' ... b2' b1' b0' |vac> (spin-major order)

    and

    ... a2' b2' a1' b1' a0' b0' |vac> (fragment-major order)

    Operators and SI vectors in LASSI use spin-major order by default. However, fragment-major
    order is sometimes more useful when computing various properties.

    Args:
        si0: ndarray of shape (ndim,*)
            Contains SI vecotr
        nelec_frs: ndarray of shape (nfrags, nroots, 2)
            Number of electrons of each spin in each fragment in each rootspace
        lroots: ndarray of shape (nfrags, nroots)
            Number of states in each fragment in each rootspace


    Returns:
        si1: ndarray of shape (ndim,*)
            si0 with permuted row signs
    '''
    nelec_rsf = np.asarray (nelec_frs).transpose (1,2,0)
    rootaddr = get_rootaddr_fragaddr (lroots)[0]
    si1 = si0.copy ()
    for iroot, nelec_sf in enumerate (nelec_rsf):
        idx = (rootaddr==iroot)
        si1[idx,:] *= fermion_spin_shuffle (nelec_sf[0], nelec_sf[1])
    return si1

# TODO: resort to spin-major order within the new vacuum! Will probably be
# necessary to make sense of excitons!
def sivec_vacuum_shuffle (si0, nelec_frs, lroots, nelec_vac=None, state=None):
    '''Define a particular number of electrons in each fragment as the vacuum,
    set the signs of the LAS basis functions accordingly and in fragment-major
    order, and return the correspondingly-modified SI vector.

    Args:
        si0: ndarray of shape (ndim,*)
            Contains SI vecotr
        nelec_frs: ndarray of shape (nfrags, nroots, 2)
            Number of electrons of each spin in each fragment in each rootspace
        lroots: ndarray of shape (nfrags, nroots)
            Number of states in each fragment in each rootspace

    Kwargs:
        nelec_vac: ndarray of shape (nfrags)
            Number of electrons (spinless) in each fragment in the new vacuum.
            Defaults to nelec_frs[:,state,:].sum (1)
        state: integer
            Index of the rootspace identified as the new vacuum. Required if
            nelec_vac is unset.

    Returns:
        si1: ndarray of shape (ndim,*)
            si0 with permuted row signs corresponding to nelec_vac electrons in
            each fragment in the vacuum and the fermion creation operators in
            fragment-major order
    '''
    assert ((nelec_vac is not None) or (state is not None))
    nfrags, nroots = nelec_frs.shape[:2]
    si1 = sivec_fermion_spin_shuffle (si0, nelec_frs, lroots)
    nelec_rf = nelec_frs.sum (2).T
    if nelec_vac is None: nelec_vac = nelec_rf[state]
    nelec_vac = np.asarray (nelec_vac)
    dparity_rf = np.remainder (nelec_rf - nelec_vac[None,:], 2)
    parity_vac = np.remainder (nelec_vac, 2)
    rootaddr = get_rootaddr_fragaddr (lroots)[0]
    for iroot, dparity_f in enumerate (dparity_rf):
        odd_frags = np.where (dparity_f)[0]
        nperms = 0
        # Remember: c2' c1' c0' |vac>
        for ifrag in odd_frags[::-1]:
            if ifrag == nfrags-1: continue
            nperms += sum (parity_vac[ifrag+1:])
        idx = (rootaddr==iroot)
        si1[idx,:] *= (1,-1)[nperms%2]
    return si1







