import numpy as np
import time
from pyscf.lib import logger, numpy_helper
from mrh.util.rdm import get_2CDM_from_2RDM
from itertools import product



def get_ontop_pair_density (ot, rho, ao, twoCDM_amo, ao2amo, deriv=0):
    r''' Pi(r) = i(r)*j(r)*k(r)*l(r)*g_ijkl / 2
               = rho[0](r)*rho[1](r) + i(r)*j(r)*k(r)*l(r)*l_ijkl / 2

        Args:
            ot : on-top pair density functional object
            rho : ndarray of shape (2,*,ngrids) 
                contains spin density [and derivatives] 
            ao : ndarray of shape (*, ngrids, nao)
                contains values of aos [and derivatives] 
            twoCDM_amo : ndarray of shape (mc.ncas, mc.ncas, mc.ncas, mc.ncas)
                contains spin-summed two-body cumulant density matrix in active space
            ao2amo : ndarray of shape (nao, ncas)
                molecular-orbital coefficients for active-space orbitals

        Kwargs:
            deriv : derivative order through which to calculate. Default is 0. 
                deriv > 1 not implemented

        Returns : ndarray of shape (ngrids) if deriv=0 and (4, ngrids) if deriv=1
            The on-top pair density and its derivatives if requested

    '''
    if deriv > 1:
        raise NotImplementedError("on-top density derivatives of order > 1")
    assert (rho.ndim == ao.ndim), "rho.shape={0}; ao.shape={1}".format (rho.shape, ao.shape)

    # Fix dimensionality of rho and ao
    if rho.ndim == 2:
        rho = rho.reshape (rho.shape[0], 1, rho.shape[1])
        ao = ao.reshape (1, ao.shape[0], ao.shape[1])

    # First cumulant and derivatives (chain rule! product rule!)
    Pi = np.zeros_like (rho[0])
    Pi[0] = rho[0,0] * rho[1,0]
    for ideriv in range(1,deriv):
        Pi[ideriv] = 2 * (rho[0,ideriv] * rho[1,0] 
                       + rho[0,0] * rho[1,ideriv])

    # Second cumulant and derivatives (chain rule! product rule!)
    # np.multiply, np.sum, and np.tensordot are linked against compiled libraries with multithreading, but np.einsum is not
    # Therefore I abandon the use of np.einsum
    # ijkl, ai, aj, ak, al -> a
    t0 = (time.clock (), time.time ())
    grid2amo = np.tensordot (ao, ao2amo, axes=1) #np.einsum ('ijk,kl->ijl', ao, ao2amo)
    gridkern = grid2amo[0,:,:,np.newaxis] * grid2amo[0,:,np.newaxis,:]  # 0ai, 0aj -> 0aij
    wrk = np.tensordot (gridkern, twoCDM_amo, axes=2)                   # 0aij, ijkl -> 0akl
    Pi[0] += (gridkern * wrk).sum ((1,2)) / 2                           # 0akl, 0akl -> 0a
    t0 = logger.timer (ot, 'otpd second cumulant value', *t0)
    for ideriv in range(1, deriv):
        # Fourfold tensor symmetry ijkl = klij = jilk = lkji & product rule -> factor of 4
        gridkern1 = grid2amo[ideriv,:,:,np.newaxis] * grid2amo[0,:,np.newaxis,:]    # 0ai, 1aj -> 1aij
        Pi[ideriv] = (gridkern1 * wrk).sum ((1,2)) * 2                              # 1akl, 0akl -> 1a  
        t0 = logger.timer (ot, 'otpd second cumulant derivative {}'.format (ideriv), *t0)

    # Unfix dimensionality of rho, ao, and Pi
    if Pi.shape[0] == 1:
        Pi = Pi.reshape (Pi.shape[1])
        rho = rho.reshape (rho.shape[0], rho.shape[2])
        ao = ao.reshape (ao.shape[1], ao.shape[2])

    return Pi


