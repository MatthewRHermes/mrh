import numpy as np
from pyscf.lib import logger
from pyscf.dft.numint import _dot_ao_dm

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
    print("using my version")
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
    if deriv > 0:
        assert (rho.shape[1] >= 4), rho.shape
        assert (ao.shape[0] >= 4), ao.shape
        for ideriv in range(1,4):
            Pi[ideriv] = rho[0,ideriv]*rho[1,0] + rho[0,0]*rho[1,ideriv]
    if deriv > 1:
        assert (rho.shape[1] >= 6), rho.shape
        assert (ao.shape[0] >= 10), ao.shape
        Pi[4] = -(rho[:,1:4].sum (0).conjugate () * rho[:,1:4].sum (0)).sum (0)
        Pi[4] /= 4.0
        Pi[4] += rho[0,0]*(rho[1,4]/4 + rho[0,5]*2)
        Pi[4] += rho[1,0]*(rho[0,4]/4 + rho[1,5]*2)
    t0 = logger.timer_debug1 (ot, 'otpd first cumulant', *t0)

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
    wrk0 = np.tensordot (gridkern[0], cascm2, axes=2)
    # r_0aij, P_ijkl -> P_0akl
    Pi[0] += (gridkern[0] * wrk0).sum ((1,2)) / 2
    # r_0aij, P_0aij -> P_0a
    t0 = logger.timer_debug1 (ot, 'otpd second cumulant 0th derivative', *t0)
    if deriv > 0:
        for ideriv in range (1, 4):
            # Fourfold tensor symmetry ijkl = klij = jilk = lkji
            # & product rule -> factor of 4
            gridkern[ideriv] = (grid2amo[ideriv,:,:,np.newaxis]
                * grid2amo[0,:,np.newaxis,:])
            # r_1ai,  r_0aj  -> r_1aij
            Pi[ideriv] += (gridkern[ideriv] * wrk0).sum ((1,2)) * 2
            # r_1aij, P_0aij -> P_1a
            t0 = logger.timer_debug1 (ot, 'otpd second cumulant 1st derivative'
                ' ({})'.format (ideriv), *t0)
    if deriv > 1: # The fifth slot is allocated to the "off-top Laplacian,"
        # i.e., nabla_(r1-r2)^2 Pi(r1,r2)|(r1=r2)
        # nabla_off^2 Pi = 1/2 d^ik_jl * ([nabla_r^2 phi_i] phi_j phi_k phi_l
        # + {1 - p_jk - p_jl}[nabla_r phi_i . nabla_r phi_j] phi_k phi_l)
        # using four-fold symmetry a lot! be careful!
        XX, YY, ZZ = 4, 7, 9
        gridkern[4]  = (grid2amo[[XX,YY,ZZ],:,:,np.newaxis].sum (0)
            * grid2amo[0,:,np.newaxis,:])
        # r_2ai, r_0aj -> r_2aij
        gridkern[4] += (grid2amo[1:4,:,:,np.newaxis]
            * grid2amo[1:4,:,np.newaxis,:]).sum (0)
        # r_1ai, r_1aj -> r_2aij
        wrk1 = np.tensordot (gridkern[1:4], cascm2, axes=2)
        # r_1aij, P_ijkl -> P_1akl
        Pi[4] += (gridkern[4] * wrk0).sum ((1,2)) / 2
        # r_2aij, P_0aij -> P_2a
        Pi[4] -= ((gridkern[1:4] + gridkern[1:4].transpose (0, 1, 3, 2))
            * wrk1).sum ((0,2,3)) / 2
        # r_1aij, P_1aij -> P_2a
        t0 = logger.timer (ot, 'otpd second cumulant off-top Laplacian', *t0)

    # Unfix dimensionality of rho, ao, and Pi
    # if Pi.shape[0] == 1:
    if rho_reshape:
        Pi = Pi.reshape (Pi.shape[1])
        rho = rho.reshape (rho.shape[0], rho.shape[2])
    if ao_reshape:
        ao = ao.reshape (ao.shape[1], ao.shape[2])

    return Pi


