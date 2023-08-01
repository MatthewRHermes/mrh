import numpy as np
from pyscf import lib, symm
from scipy import linalg
from mrh.my_pyscf.mcscf.lasci import get_space_info

def decompose_sivec_by_rootspace (las, si, ci=None):
    '''Decompose a set of LASSI vectors as

    si[i,:] = +sqrt(space_weights[P])*state_coeffs[P][a,:]

    Where "i" indexes the "a"th state in rootspace "P"'''
    if ci is None: ci=las.ci
    if si.ndim==1: si = si[:,None]
    lroots = np.array ([[1 if c.ndim<3 else c.shape[0]
                         for c in ci_r]
                        for ci_r in ci])
    nstates = np.product (lroots, axis=0)
    jj = np.cumsum (nstates)
    ii = jj - nstates
    nspaces = las.nroots
    nroots = si.shape[1]
    space_coeffs = np.empty ((nspaces, nroots), dtype=si.dtype)
    state_coeffs = []
    for space, (i, j) in enumerate (zip (ii, jj)):
        space_coeffs[space] = linalg.norm (si[i:j,:], axis=0)
        idx = space_coeffs[space]>0
        state_coeffs.append (si[i:j,:].copy ())
        state_coeffs[-1][:,idx] /= space_coeffs[space][idx]
    return space_coeffs**2, state_coeffs

def make_sdm1 (sivec, lroots, site):
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
    nroots = sivec.size // np.prod (lroots)
    sivec = np.asfortranarray (sivec)
    sivec = sivec.reshape (list(lroots)+[nroots,], order='F')
    idx = [site,nsites]
    if site<nsites-1: idx = list(range(site+1,nsites)) + idx
    if site>0: idx = list(range(site)) + idx
    sivec = sivec.transpose (*idx).reshape (-1, lroots[site],nroots)
    return lib.einsum ('api,aqi->ipq', sivec.conj(), sivec)
    
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
        log.info ("Remaining %d ONVs in rootspace %d have combined average weight = %e",
                  nprods-ix-1, iroot, running_weight)
    else:
        log.info ("All %d ONVs in rootspace %d accounted for", nprods, iroot)
    return


def analyze (las, si, state=0, print_all_but=1e-8):
    '''Print out analysis of LASSI result in terms of average quantum numbers
    and density matrix analyses of the lroots in each rootspace

    Args:
        las: instance of :class:`LASCINoSymm`
        si: ndarray of shape (nstates,nstates)

    Kwargs:
        state: integer or index array
            indicates which columns of si to analyze. Indicated columns are
            averaged together for the lroots analysis
        print_all_but: continue density-matrix analysis printouts until all
            but this weight of the wave function(s) is accounted for. Set
            to zero to print everything.
    '''
    log = lib.logger.new_logger (las, las.verbose)
    log.info ("Analyzing LASSI vectors for states = %s",str(state))

    log.info ("Average quantum numbers:")
    space_weights, state_coeffs = decompose_sivec_by_rootspace (las, si[:,state])
    states = np.atleast_1d (state)
    nelelas = np.array ([sum (n) for n in las.nelecas_sub])
    c, m, smults = get_rootspace_central_moment (las, space_weights)
    neleca = .5*(nelelas[None,:]-c+m)
    nelecb = .5*(nelelas[None,:]-c-m)
    for na, nb, s, st in zip (neleca, nelecb, smults, states):
        log.info ("State %d:", st)
        log.info ("Neleca = %s", str (na))
        log.info ("Nelecb = %s", str (nb))
        log.info ("Smult = %s", str (s))

    log.info (("Analyzing rootspace fragment density matrices for LASSI "
               "states %s averaged together"), str (states))
    log.info ("Continue until 1-%e of wave function(s) accounted for", print_all_but)
    nstates = len (states)
    avg_weights = space_weights.sum (1) / nstates
    lroots = np.array ([[1 if c.ndim<3 else c.shape[0]
                         for c in ci_r]
                        for ci_r in las.ci]).T
    running_weight = 1
    c, m, s, w = get_space_info (las)
    fmt_str = " {:4s}  {:>7s}  {:>4s}  {:>3s}  {:>6s}  {:11s}  {:>8s}"
    header = fmt_str.format ("Frag", "Nelec", "2S+1", "Ir", "<n>", "Max(weight)", "Entropy")
    fmt_str = " {:4d}  {:>7s}  {:>4d}  {:>3s}  {:6.3f}  {:>11.4f}  {:8f}"
    for ix, iroot in enumerate (np.argsort (-avg_weights)):
        log.info ("Rootspace %d with averaged weight %9.3e", iroot, avg_weights[iroot])
        log.info (header)
        for ifrag in range (las.nfrags):
            sdm = make_sdm1 (state_coeffs[iroot], lroots[iroot], ifrag).sum (0) / nstates
            dens = sdm.diagonal ()
            navg = np.dot (np.arange (len (dens)), dens)
            maxw = np.amax (dens)
            evals, evecs = linalg.eigh (sdm)
            evals = evals[evals>0]
            entr = abs(np.dot (evals, np.log (evals)))
            nelec = "{}a+{}b".format ((nelelas[ifrag]-c[iroot,ifrag]+m[iroot,ifrag])//2,
                                      (nelelas[ifrag]-c[iroot,ifrag]-m[iroot,ifrag])//2)
            ir = symm.irrep_id2name (las.mol.groupname, w[iroot][ifrag])
            log.info (fmt_str.format (ifrag, nelec, s[iroot][ifrag], ir, navg, maxw, entr))
        log.info ("Printing wave function(s) in rootspace %d:", iroot)
        _print_states (log, iroot, space_weights[iroot], state_coeffs[iroot], lroots[iroot],
                       print_all_but=print_all_but)
        running_weight -= avg_weights[iroot]
        if running_weight < print_all_but: break

    if ix+1<las.nroots:
        log.info ("Remaining %d rootspaces have combined weight = %e",
                  las.nroots-ix-1, running_weight)
    else:
        log.info ("All %d rootspaces accounted for", las.nroots)

    return


