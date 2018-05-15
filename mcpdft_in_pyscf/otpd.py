import numpy as np
from mrh.util.rdm import get_2CDM_from_2RDM
from itertools import product



def get_ontop_pair_density (rho, ao, twoCDM_amo, ao2amo, deriv=0):
    r''' Pi(r) = i(r)*j(r)*k(r)*l(r)*g_ijkl 
               = rho[0](r)*rho[1](r) + i(r)*j(r)*k(r)*l(r)*l_ijkl

        Args:
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
        Pi[ideriv] = (rho[0,ideriv] * rho[1,0] 
                    + rho[0,0] * rho[1,ideriv])

    # Second cumulant and derivatives (chain rule! product rule!)
    grid2amo = np.einsum ('ijk,kl->ijl', ao, ao2amo)
    Pi[0] += np.einsum ('ijkl,ai,aj,ak,al->a', twoCDM_amo, grid2amo[0], grid2amo[0], grid2amo[0], grid2amo[0])
    for ideriv in range(1, deriv):
        Pi[ideriv] += np.einsum ('ijkl,ai,aj,ak,al->a', twoCDM_amo, grid2amo[ideriv], grid2amo[0], grid2amo[0], grid2amo[0])
        Pi[ideriv] += np.einsum ('ijkl,ai,aj,ak,al->a', twoCDM_amo, grid2amo[0], grid2amo[ideriv], grid2amo[0], grid2amo[0])
        Pi[ideriv] += np.einsum ('ijkl,ai,aj,ak,al->a', twoCDM_amo, grid2amo[0], grid2amo[0], grid2amo[ideriv], grid2amo[0])
        Pi[ideriv] += np.einsum ('ijkl,ai,aj,ak,al->a', twoCDM_amo, grid2amo[0], grid2amo[0], grid2amo[0], grid2amo[ideriv])

    # Unfix dimensionality of rho, ao, and Pi
    if Pi.shape[0] == 1:
        Pi = Pi.reshape (Pi.shape[1])
        rho = rho.reshape (rho.shape[0], rho.shape[2])
        ao = ao.reshape (ao.shape[1], ao.shape[2])

    return Pi


