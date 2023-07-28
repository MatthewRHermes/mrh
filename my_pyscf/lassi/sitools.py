import numpy as np
from pyscf import lib
from scipy import linalg
from mrh.my_pyscf.mcscf.lasci import get_space_info


def decompose_sivec_by_rootspace(las, si, ci=None):
    '''Decompose a set of LASSI vectors as

    si[i,:] = +sqrt(space_weights[P])*state_coeffs[P][a,:]

    Where "i" indexes the "a"th state in rootspace "P"'''
    if ci is None: ci = las.ci
    lroots = np.array([[1 if c.ndim < 3 else c.shape[0]
                        for c in ci_r]
                       for ci_r in ci])
    nstates = np.product(lroots, axis=0)
    jj = np.cumsum(nstates)
    ii = jj - nstates
    nspaces = las.nroots
    nroots = si.shape[1]
    space_coeffs = np.empty((nspaces, nroots), dtype=si.dtype)
    state_coeffs = []
    for space, (i, j) in enumerate(zip(ii, jj)):
        space_coeffs[space] = linalg.norm(si[i:j, :], axis=0)
        idx = space_coeffs[space] > 0
        state_coeffs.append(si[i:j, :].copy())
        state_coeffs[-1][:, idx] /= space_coeffs[space][idx]
    return space_coeffs ** 2, state_coeffs


def make_sdm1(sivec, lroots, site):
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
    nsites = len(lroots)
    nroots = sivec.size // np.prod(lroots)
    sivec = np.asfortranarray(sivec)
    sivec = sivec.reshape(list(lroots) + [nroots, ], order='F')
    idx = [site, nsites]
    if site < nsites - 1: idx = list(range(site + 1, nsites)) + idx
    if site > 0: idx = list(range(site)) + idx
    sivec = sivec.transpose(*idx).reshape(-1, lroots[site], nroots)
    return lib.einsum('api,aqi->ipq', sivec.conj(), sivec)


def get_rootspace_central_moment(las, space_weights, n=1):
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
    if space_weights.ndim < 2:
        space_weights = space_weights[:, None]
    nvec = space_weights.shape[1]
    charges, spins, smults, wfnsyms = get_space_info(las)
    c1 = np.dot(charges.T, space_weights).T
    m1 = np.dot(spins.T, space_weights).T
    s1 = np.dot(smults.T, space_weights).T
    if n == 1:
        return c1, m1, s1
    c2 = lib.einsum('frv,rv->vf', np.power(charges.T[:, :, None] - c1.T[:, None, :], n), space_weights)
    m2 = lib.einsum('frv,rv->vf', np.power(spins.T[:, :, None] - m1.T[:, None, :], n), space_weights)
    s2 = lib.einsum('frv,rv->vf', np.power(smults.T[:, :, None] - s1.T[:, None, :], n), space_weights)
    return c2, m2, s2




