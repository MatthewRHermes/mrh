import numpy as np
from pyscf.lib import logger
from pyscf.dft.numint import _dot_ao_dm

def _grid_ao2mo (mol, ao, mo_coeff, non0tab=None, shls_slice=None,
        ao_loc=None):
    '''ao[deriv,grid,AO].mo_coeff[AO,MO]->mo[deriv,grid,MO]
    ASSUMES that ao is in data layout (deriv,AO,grid) in row-major order!
    mo is returned in data layout (deriv,MO,grid) in row-major order '''
    nderiv, ngrid, _ = ao.shape
    nmo = mo_coeff.shape[-1]
    mo = np.empty ((nderiv,nmo,ngrid), dtype=mo_coeff.dtype, order='C')
    mo = mo.transpose (0,2,1)
    if shls_slice is None: shls_slice = (0, mol.nbas)
    if ao_loc is None: ao_loc = mol.ao_loc_nr ()
    for ideriv in range (nderiv):
        ao_i = ao[ideriv,:,:]
        mo[ideriv] = _dot_ao_dm (mol, ao_i, mo_coeff, non0tab, shls_slice,
            ao_loc, out=mo[ideriv])
    return mo

def get_ontop_pair_density (ot, rho, ao, cascm2, mo_cas, deriv=0,
        non0tab=None):
    r'''Compute the on-top pair density and its derivatives on a grid:

    Pi(r) = i(r)*j(r)*k(r)*l(r)*d_ijkl / 2
          = rho[0](r)*rho[1](r) + i(r)*j(r)*k(r)*l(r)*l_ijkl / 2

    Args:
        ot : on-top pair density functional object
        rho : ndarray of shape (2,*,ngrids)
            Contains spin-separated density [and derivatives]. The dm1s
            underlying these densities must correspond to the dm1s/dm1
            in the expression for cascm2 below.
        ao : ndarray of shape (*, ngrids, nao)
            contains values of aos [and derivatives]
        cascm2 : ndarray of shape [ncas,]*4
            contains spin-summed two-body cumulant density matrix in an
            active-orbital basis given by mo_cas:
                cm2[u,v,x,y] = dm2[u,v,x,y] - dm1[u,v]*dm1[x,y]
                               + dm1s[0][u,y]*dm1s[0][x,v]
                               + dm1s[1][u,y]*dm1s[1][x,v]
            where dm1 = dm1s[0] + dm1s[1]. The cumulant (cm2) has no
            nonzero elements for any index outside the active space,
            unlike the density matrix (dm2), which formally has elements
            involving uncorrelated, doubly-occupied ``core'' orbitals
            which are not usually computed explicitly:
                dm2[i,i,u,v] = dm2[u,v,i,i] = 2*dm1[u,v]
                dm2[u,i,i,v] = dm2[i,v,u,i] = -dm1[u,v]
        mo_cas : ndarray of shape (nao, ncas)
            molecular-orbital coefficients for active-space orbitals

    Kwargs:
        deriv : derivative order through which to calculate.
            deriv > 1 not implemented
        non0tab : as in pyscf.dft.gen_grid and pyscf.dft.numint

    Returns : ndarray of shape (*,ngrids)
        The on-top pair density and its derivatives if requested
        deriv = 0 : value (1d array)
        deriv = 1 : value, d/dx, d/dy, d/dz
        deriv = 2 : value, d/dx, d/dy, d/dz, d^2/d|r1-r2|^2_(r1=r2)
    '''
    # Fix dimensionality of rho and ao
    rho_reshape = False
    ao_reshape = False
    if rho.ndim == 2:
        rho_reshape = True
        rho = rho.reshape(rho.shape[0], 1, rho.shape[1])
    if ao.ndim == 2:
        ao_reshape = True
        ao = ao.reshape(1, ao.shape[0], ao.shape[1])

    # First cumulant and derivatives (chain rule! product rule!)
    t0 = (logger.process_clock (), logger.perf_counter ())
    Pi_shape = ((1,4,5)[deriv], rho.shape[-1])
    Pi = np.zeros(Pi_shape, dtype=rho.dtype)
    Pi[0] = rho[0,0] * rho[1,0]
    t0 = logger.timer (ot, 'otpd first cumulant', *t0)

    # Second cumulant and derivatives (chain rule! product rule!)
    # dot, tensordot, and sum are hugely faster than np.einsum
    # but whether or when they actually multithread is unclear
    # Update 05/11/2020: ao is actually stored in row-major order
    # = (deriv,AOs,grids).
    grid2amo = _grid_ao2mo (ot.mol, ao, mo_cas, non0tab=non0tab)
    t0 = logger.timer (ot, 'otpd ao2mo', *t0)
    gridkern = np.zeros (grid2amo.shape + (grid2amo.shape[2],),
        dtype=grid2amo.dtype)
    gridkern[0] = grid2amo[0,:,:,np.newaxis] * grid2amo[0,:,np.newaxis,:]
    # r_0ai,  r_0aj  -> r_0aij
    t0 = logger.timer(ot, 'before tensorot', *t0)
    wrk0 = np.tensordot (gridkern[0], cascm2, axes=2)
    # r_0aij, P_ijkl -> P_0akl
    Pi[0] += (gridkern[0] * wrk0).sum ((1,2)) / 2
    # r_0aij, P_0aij -> P_0a
    t0 = logger.timer(ot, 'otpd second cumulant 0th derivative', *t0)
    # Unfix dimensionality of rho, ao, and Pi
    # if Pi.shape[0] == 1:
    if rho_reshape:
        Pi = Pi.reshape (Pi.shape[1])
        rho = rho.reshape (rho.shape[0], rho.shape[2])
    if ao_reshape:
        ao = ao.reshape (ao.shape[1], ao.shape[2])

    return Pi

