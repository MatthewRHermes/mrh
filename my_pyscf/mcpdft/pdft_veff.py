from pyscf.lib import logger
from mrh.my_pyscf.mcpdft.otpd import get_ontop_pair_density
import numpy as np
import time

def kernel (ot, oneCDMs, twoCDM_amo, ao2amo, max_memory=20000, hermi=1, veff2_mo=None):
    ''' Get the 1- and 2-body effective potential from MC-PDFT. Eventually I'll be able to specify
        mo slices for the 2-body part

        Args:
            ot : an instance of otfnal class
            oneCDMs : ndarray of shape (2, nao, nao)
                containing spin-separated one-body density matrices
            twoCDM_amo : ndarray of shape (ncas, ncas, ncas, ncas)
                containing spin-summed two-body cumulant density matrix in an active space
            ao2amo : ndarray of shape (nao, ncas)
                containing molecular orbital coefficients for active-space orbitals

        Kwargs:
            max_memory : int or float
                maximum cache size in MB
                default is 20000
            hermi : int
                1 if 1CDMs are assumed hermitian, 0 otherwise

        Returns : float
            The MC-PDFT on-top exchange-correlation energy

    '''
    if veff2_mo is not None:
        raise NotImplementedError ('Molecular orbital slices for the two-body part')
    ni, xctype, dens_deriv = ot._numint, ot.xctype, ot.dens_deriv
    norbs_ao = ao2amo.shape[0]

    veff1 = np.zeros_like (oneCDMs[0])
    veff2 = np.zeros ((norbs_ao, norbs_ao, norbs_ao, norbs_ao), dtype=veff1.dtype)

    t0 = (time.clock (), time.time ())
    make_rho = tuple (ni._gen_rho_evaluator (ot.mol, oneCDMs[i,:,:], hermi) for i in range(2))
    for ao, mask, weight, coords in ni.block_loop (ot.mol, ot.grids, norbs_ao, dens_deriv, max_memory):
        rho = np.asarray ([m[0] (0, ao, mask, xctype) for m in make_rho])
        t0 = logger.timer (ot, 'untransformed density', *t0)
        Pi = get_ontop_pair_density (ot, rho, ao, oneCDMs, twoCDM_amo, ao2amo, dens_deriv)
        t0 = logger.timer (ot, 'on-top pair density calculation', *t0)
        veff1 += ot.get_veff_1body (rho, Pi, ao, weight)
        t0 = logger.timer (ot, '1-body effective potential calculation', *t0)
        veff2 += ot.get_veff_2body (rho, Pi, ao, weight)
        t0 = logger.timer (ot, '2-body effective potential calculation', *t0)
    return veff1, veff2

def get_veff_1body (otfnal, rho, Pi, ao, weight, **kwargs):
    r''' get the derivatives dEot / dDpq

    Args:
        rho : ndarray of shape (2,*,ngrids)
            containing spin-density [and derivatives]
        Pi : ndarray with shape (*,ngrids)
            containing on-top pair density [and derivatives]
        ao : ndarray of shape (*,ngrids,nao)
            contains values and derivatives of nao
        weight : ndarray of shape (ngrids)
            containing numerical integration weights

    Returns : ndarray of shape (nao,nao)
        The 1-body effective potential corresponding to this on-top pair density
        exchange-correlation functional, in the atomic-orbital basis.
        In PDFT this functional is always spin-symmetric
    '''

    nderiv = rho.shape[1]
    w = weight[None,:]
    ao1 = ao[:,:,None]
    ao2 = ao[:,None,:]
    kern = otfnal.get_dEot_drho (rho, Pi, **kwargs) * weight[None,:]

    # Broadcasting setup. First two indices in every case are deriv and grid
    kern = kern[:,:,None,None]
    ao1 = ao[:,:,:,None]
    ao2 = ao[:,:,None,:]

    # Zeroth derivative
    veff = (kern[0] * ao1[0] * ao2[0]).sum (0)
    # First derivatives. kern[ideriv] is a chain rule cofactor, not a product rule cofactor
    for ideriv in range (1,min(nderiv,4)):
        veff_ideriv = (kern[ideriv] * ao1[ideriv] * ao2[0]).sum (0)
        veff += veff_ideriv + veff_ideriv.T
    return veff

def get_veff_2body (otfnal, rho, Pi, ao, weight, **kwargs):
    r''' get the derivatives dEot / dPpqrs

    Args:
        rho : ndarray of shape (2,*,ngrids)
            containing spin-density [and derivatives]
        Pi : ndarray with shape (*,ngrids)
            containing on-top pair density [and derivatives]
        ao : ndarray of shape (*,ngrids,nao) OR list of ndarrays of shape (*,ngrids,*)
            values and derivatives of atomic or molecular orbitals in which
            space to calculate the 2-body veff
            If a list of length 4, the corresponding set of eri-like elements are returned
        weight : ndarray of shape (ngrids)
            containing numerical integration weights

    Returns : eri-like ndarray
        The two-body effective potential corresponding to this on-top pair density
        exchange-correlation functional or elements thereof, in the provided basis.
    '''

    if isinstance (ao, np.ndarray) and ao.ndim == 3:
        ao = [ao,ao,ao,ao]
    else:
        raise NotImplementedError ('orbital subsets and fast evaluation in get_veff_2body')
    #elif isinstance (ao, np.ndarray) and ao.ndim == 4:
    #    ao = [a for a in ao]

    nderiv = Pi.shape[0]
    kern = otfnal.get_dEot_dPi (rho, Pi, **kwargs) * weight[None,:]

    # Zeroth derivative
    veff = np.einsum ('g,gp,gq,gr,gs->pqrs',kern[0],ao[0][0],ao[1][0],ao[2][0],ao[3][0])

    # First derivatives
    veff_deriv = np.einsum ('dg,dgp,gq,gr,gs->pqrs',kern[1:4],ao[0][1:4],ao[1][0],ao[2][0],ao[3][0])
    veff += veff_deriv
    veff += veff_deriv.transpose (1,0,3,2)
    veff += veff_deriv.transpose (2,3,0,1)
    veff += veff_deriv.transpose (3,2,1,0)
    return veff 



