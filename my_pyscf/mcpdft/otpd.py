import numpy as np
import time
from scipy import linalg
from pyscf.lib import logger
from pyscf.lib import einsum as einsum_threads
from pyscf.dft.numint import _dot_ao_dm
from mrh.util.rdm import get_2CDM_from_2RDM, get_2RDM_from_2CDM
from mrh.util.basis import represent_operator_in_basis
from itertools import product
from os import path

def _grid_ao2mo (mol, ao, mo_coeff, non0tab=None, shls_slice=None, ao_loc=None):
    ''' ao[deriv,grid,AO].mo_coeff[AO,MO]->mo[deriv,grid,MO]
    ASSUMES that ao is in data layout (deriv,AO,grid) in row-major order!
    mo is returned in data layout (deriv,MO,grid) in row-major order '''
    nderiv, ngrid, nao = ao.shape
    nmo = mo_coeff.shape[-1]
    mo = np.empty ((nderiv,nmo,ngrid), dtype=mo_coeff.dtype, order='C').transpose (0,2,1)
    if shls_slice is None: shls_slice = (0, mol.nbas)
    if ao_loc is None: ao_loc = mol.ao_loc_nr ()
    for ideriv in range (nderiv):
        ao_i = ao[ideriv,:,:]
        mo[ideriv] = _dot_ao_dm (mol, ao_i, mo_coeff, non0tab, shls_slice, ao_loc, out=mo[ideriv])
    return mo 


def get_ontop_pair_density (ot, rho, ao, oneCDMs, twoCDM_amo, ao2amo, deriv=0, non0tab=None):
    r''' Pi(r) = i(r)*j(r)*k(r)*l(r)*g_ijkl / 2
               = rho[0](r)*rho[1](r) + i(r)*j(r)*k(r)*l(r)*l_ijkl / 2

        Args:
            ot : on-top pair density functional object
            rho : ndarray of shape (2,*,ngrids) 
                contains spin density [and derivatives] 
            ao : ndarray of shape (*, ngrids, nao)
                contains values of aos [and derivatives] 
            oneCDMs : ndarray of shape (2, nao, nao)
                contains spin-separated 1-RDM
            twoCDM_amo : ndarray of shape (mc.ncas, mc.ncas, mc.ncas, mc.ncas)
                contains spin-summed two-body cumulant density matrix in active space
            ao2amo : ndarray of shape (nao, ncas)
                molecular-orbital coefficients for active-space orbitals

        Kwargs:
            deriv : derivative order through which to calculate. Default is 0. 
                deriv > 1 not implemented
            non0tab : as in pyscf.dft.gen_grid and pyscf.dft.numint

        Returns : ndarray of shape (*,ngrids)
            The on-top pair density and its derivatives if requested
            deriv = 0 : value (1d array)
            deriv = 1 : value, d/dx, d/dy, d/dz
            deriv = 2 : value, d/dx, d/dy, d/dz, d^2/d|r1-r2|^2_(r1=r2)
            

    '''
    assert (rho.ndim == ao.ndim), "rho.shape={0}; ao.shape={1}".format (rho.shape, ao.shape)

    # Fix dimensionality of rho and ao
    if rho.ndim == 2:
        rho = rho.reshape (rho.shape[0], 1, rho.shape[1])
        ao = ao.reshape (1, ao.shape[0], ao.shape[1])

    # Debug code for ultra-slow, ultra-high-memory but very safe implementation
    if ot.verbose > logger.DEBUG:
        logger.debug (ot, 'Warning: memory-intensive cacheing of full 2RDM for testing '
            'purposes initiated; reduce verbosity to increase speed and memory efficiency')
        twoRDM = represent_operator_in_basis (twoCDM_amo, ao2amo.conjugate ().T)
        twoRDM = get_2RDM_from_2CDM (twoRDM, oneCDMs)

    # First cumulant and derivatives (chain rule! product rule!)
    t0 = (time.clock (), time.time ())
    Pi = np.zeros_like (rho[0])
    Pi[0] = rho[0,0] * rho[1,0]
    if deriv > 0:
        assert (rho.shape[1] >= 4), rho.shape
        assert (ao.shape[0] >= 4), ao.shape
        for ideriv in range(1,4):
            Pi[ideriv] = rho[0,ideriv]*rho[1,0] + rho[0,0]*rho[1,ideriv]
    if deriv > 1:
        assert (rho.shape[1] >= 6), rho.shape
        assert (ao.shape[0] >= 10), ao.shape
        Pi[4] = -(rho[:,1:4].sum (0).conjugate () * rho[:,1:4].sum (0)).sum (0) / 4
        Pi[4] += rho[0,0]*(rho[1,4]/4 + rho[0,5]*2) 
        Pi[4] += rho[1,0]*(rho[0,4]/4 + rho[1,5]*2)
    t0 = logger.timer_debug1 (ot, 'otpd first cumulant', *t0)

    # Second cumulant and derivatives (chain rule! product rule!)
    # dot, tensordot, and sum are hugely faster than np.einsum 
    # but whether or when they actually multithread is unclear
    # Update 05/11/2020: ao is actually stored in row-major order
    # = (deriv,AOs,grids).
    #grid2amo_ref = np.tensordot (ao, ao2amo, axes=1) #np.einsum ('ijk,kl->ijl', ao, ao2amo)
    grid2amo = _grid_ao2mo (ot.mol, ao, ao2amo, non0tab=non0tab)
    t0 = logger.timer (ot, 'otpd ao2mo', *t0)
    gridkern = np.zeros (grid2amo.shape + (grid2amo.shape[2],), dtype=grid2amo.dtype)
    gridkern[0] = grid2amo[0,:,:,np.newaxis] * grid2amo[0,:,np.newaxis,:]  # r_0ai,  r_0aj  -> r_0aij
    wrk0 = np.tensordot (gridkern[0], twoCDM_amo, axes=2)                  # r_0aij, P_ijkl -> P_0akl
    Pi[0] += (gridkern[0] * wrk0).sum ((1,2)) / 2                          # r_0aij, P_0aij -> P_0a
    t0 = logger.timer_debug1 (ot, 'otpd second cumulant 0th derivative', *t0)
    if ot.verbose > logger.DEBUG:
        logger.debug (ot, 'Warning: slow einsum-based testing calculation of Pi initiated; '
            'reduce verbosity to increase speed and memory efficiency')
        test_Pi = np.einsum ('ijkl,ai,aj,ak,al->a', twoRDM, ao[0], ao[0], ao[0], ao[0]) / 2
        logger.debug (ot, "Pi, |tensordot_formula - einsum_formula| = %s", linalg.norm (Pi[0] - test_Pi)) 
        t0 = logger.timer (ot, 'otpd 0th derivative debug'.format (ideriv), *t0)
    if deriv > 0:
        for ideriv in range (1, 4):
            # Fourfold tensor symmetry ijkl = klij = jilk = lkji & product rule -> factor of 4
            gridkern[ideriv] = grid2amo[ideriv,:,:,np.newaxis] * grid2amo[0,:,np.newaxis,:]    # r_1ai,  r_0aj  -> r_1aij
            Pi[ideriv] += (gridkern[ideriv] * wrk0).sum ((1,2)) * 2                            # r_1aij, P_0aij -> P_1a  
            t0 = logger.timer_debug1 (ot, 'otpd second cumulant 1st derivative ({})'.format (ideriv), *t0)
            if ot.verbose > logger.DEBUG:
                logger.debug (ot, 'Warning: slow einsum-based testing calculation of Pi\'s first derivatives initiated; '
                    'reduce verbosity to increase speed and memory efficiency')
                test_Pi  = np.einsum ('ijkl,ai,aj,ak,al->a', twoRDM, ao[ideriv], ao[0], ao[0], ao[0]) / 2
                test_Pi += np.einsum ('ijkl,aj,ai,ak,al->a', twoRDM, ao[ideriv], ao[0], ao[0], ao[0]) / 2
                test_Pi += np.einsum ('ijkl,ak,ai,aj,al->a', twoRDM, ao[ideriv], ao[0], ao[0], ao[0]) / 2
                test_Pi += np.einsum ('ijkl,al,ai,aj,ak->a', twoRDM, ao[ideriv], ao[0], ao[0], ao[0]) / 2
                logger.debug (ot, "Pi derivative, |tensordot_formula - einsum_formula| = %s", linalg.norm (Pi[ideriv] - test_Pi)) 
                t0 = logger.timer (ot, 'otpd 1st derivative ({}) debug'.format (ideriv), *t0)
    if deriv > 1: # The fifth slot is allocated to the "off-top Laplacian," i.e., nabla_(r1-r2)^2 Pi(r1,r2)|(r1=r2) 
        # nabla_off^2 Pi = 1/2 d^ik_jl * ([nabla_r^2 phi_i] phi_j phi_k phi_l + {1 - p_jk - p_jl}[nabla_r phi_i . nabla_r phi_j] phi_k phi_l)
        # using four-fold symmetry a lot! be careful!
        if ot.verbose > logger.DEBUG:
            test2_Pi = Pi[4].copy ()
        XX, YY, ZZ = 4, 7, 9
        gridkern[4]  = grid2amo[[XX,YY,ZZ],:,:,np.newaxis].sum (0) * grid2amo[0,:,np.newaxis,:]    # r_2ai, r_0aj -> r_2aij
        gridkern[4] += (grid2amo[1:4,:,:,np.newaxis] * grid2amo[1:4,:,np.newaxis,:]).sum (0)       # r_1ai, r_1aj -> r_2aij
        wrk1 = np.tensordot (gridkern[1:4], twoCDM_amo, axes=2)                                    # r_1aij, P_ijkl -> P_1akl
        Pi[4] += (gridkern[4] * wrk0).sum ((1,2)) / 2                                              # r_2aij, P_0aij -> P_2a
        Pi[4] -= ((gridkern[1:4] + gridkern[1:4].transpose (0, 1, 3, 2)) * wrk1).sum ((0,2,3)) / 2 # r_1aij, P_1aij -> P_2a
        t0 = logger.timer (ot, 'otpd second cumulant off-top Laplacian', *t0)
        if ot.verbose > logger.DEBUG:
            logger.debug (ot, 'Warning: slow einsum-based testing calculation of Pi\'s second derivatives initiated; '
                    'reduce verbosity to increase speed and memory efficiency')
            X, Y, Z = 1, 2, 3
            test_Pi  = np.einsum ('ijkl,ai,aj,ak,al->a', twoRDM, ao[XX], ao[0], ao[0], ao[0]) / 2
            test_Pi += np.einsum ('ijkl,ai,aj,ak,al->a', twoRDM, ao[YY], ao[0], ao[0], ao[0]) / 2
            test_Pi += np.einsum ('ijkl,ai,aj,ak,al->a', twoRDM, ao[ZZ], ao[0], ao[0], ao[0]) / 2
            test_Pi += np.einsum ('ijkl,ai,aj,ak,al->a', twoRDM, ao[X], ao[X], ao[0], ao[0]) / 2
            test_Pi += np.einsum ('ijkl,ai,aj,ak,al->a', twoRDM, ao[Y], ao[Y], ao[0], ao[0]) / 2
            test_Pi += np.einsum ('ijkl,ai,aj,ak,al->a', twoRDM, ao[Z], ao[Z], ao[0], ao[0]) / 2
            test_Pi -= np.einsum ('ijkl,ai,aj,ak,al->a', twoRDM, ao[X], ao[0], ao[X], ao[0]) / 2
            test_Pi -= np.einsum ('ijkl,ai,aj,ak,al->a', twoRDM, ao[Y], ao[0], ao[Y], ao[0]) / 2
            test_Pi -= np.einsum ('ijkl,ai,aj,ak,al->a', twoRDM, ao[Z], ao[0], ao[Z], ao[0]) / 2
            test_Pi -= np.einsum ('ijkl,ai,aj,ak,al->a', twoRDM, ao[X], ao[0], ao[0], ao[X]) / 2
            test_Pi -= np.einsum ('ijkl,ai,aj,ak,al->a', twoRDM, ao[Y], ao[0], ao[0], ao[Y]) / 2
            test_Pi -= np.einsum ('ijkl,ai,aj,ak,al->a', twoRDM, ao[Z], ao[0], ao[0], ao[Z]) / 2
            logger.debug (ot, 'Pi off-top Laplacian, |tensordot formula - einsum_formula| = %s', linalg.norm (Pi[4] - test_Pi))

            test2_Pi += np.einsum ('ijkl,ai,aj,ak,al->a', twoCDM_amo, grid2amo[XX], grid2amo[0], grid2amo[0], grid2amo[0]) / 2
            test2_Pi += np.einsum ('ijkl,ai,aj,ak,al->a', twoCDM_amo, grid2amo[YY], grid2amo[0], grid2amo[0], grid2amo[0]) / 2
            test2_Pi += np.einsum ('ijkl,ai,aj,ak,al->a', twoCDM_amo, grid2amo[ZZ], grid2amo[0], grid2amo[0], grid2amo[0]) / 2
            test2_Pi += np.einsum ('ijkl,ai,aj,ak,al->a', twoCDM_amo, grid2amo[X], grid2amo[X], grid2amo[0], grid2amo[0]) / 2
            test2_Pi += np.einsum ('ijkl,ai,aj,ak,al->a', twoCDM_amo, grid2amo[Y], grid2amo[Y], grid2amo[0], grid2amo[0]) / 2
            test2_Pi += np.einsum ('ijkl,ai,aj,ak,al->a', twoCDM_amo, grid2amo[Z], grid2amo[Z], grid2amo[0], grid2amo[0]) / 2
            test2_Pi -= np.einsum ('ijkl,ai,aj,ak,al->a', twoCDM_amo, grid2amo[X], grid2amo[0], grid2amo[X], grid2amo[0]) / 2
            test2_Pi -= np.einsum ('ijkl,ai,aj,ak,al->a', twoCDM_amo, grid2amo[Y], grid2amo[0], grid2amo[Y], grid2amo[0]) / 2
            test2_Pi -= np.einsum ('ijkl,ai,aj,ak,al->a', twoCDM_amo, grid2amo[Z], grid2amo[0], grid2amo[Z], grid2amo[0]) / 2
            test2_Pi -= np.einsum ('ijkl,ai,aj,ak,al->a', twoCDM_amo, grid2amo[X], grid2amo[0], grid2amo[0], grid2amo[X]) / 2
            test2_Pi -= np.einsum ('ijkl,ai,aj,ak,al->a', twoCDM_amo, grid2amo[Y], grid2amo[0], grid2amo[0], grid2amo[Y]) / 2
            test2_Pi -= np.einsum ('ijkl,ai,aj,ak,al->a', twoCDM_amo, grid2amo[Z], grid2amo[0], grid2amo[0], grid2amo[Z]) / 2
            logger.debug (ot, 'Pi off-top Laplacian, testing second cumulant only |tensordot formula - einsum_formula| = %s', linalg.norm (Pi[4] - test2_Pi))
            
            t0 = logger.timer (ot, 'otpd off-top Laplacian debug', *t0)

    # Unfix dimensionality of rho, ao, and Pi
    if Pi.shape[0] == 1:
        Pi = Pi.reshape (Pi.shape[1])
        rho = rho.reshape (rho.shape[0], rho.shape[2])
        ao = ao.reshape (ao.shape[1], ao.shape[2])

    return Pi


