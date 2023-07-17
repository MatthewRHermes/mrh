import numpy as np
from pyscf import lib
from scipy import linalg

def decompose_sivec_by_rootspace (las, si, ci=None):
    '''Decompose a set of LASSI vectors as

    si[i,:] = +sqrt(space_weights[P])*state_coeffs[P][a,:]

    Where "i" indexes the "a"th state in rootspace "P"'''
    if ci is None: ci=las.ci
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
    



