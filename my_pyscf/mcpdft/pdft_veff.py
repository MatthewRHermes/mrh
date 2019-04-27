from pyscf.lib import logger
from mrh.my_pyscf.mcpdft.otpd import get_ontop_pair_density
from scipy import linalg
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

def get_veff_1body (otfnal, rho, Pi, ao, weight, kern=None, **kwargs):
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

    Kwargs:
        kern : ndarray of shape (*,ngrids)
            the derivative of the on-top potential with respect to density (vrho)
            If not provided, it is calculated.

    Returns : ndarray of shape (nao,nao)
        The 1-body effective potential corresponding to this on-top pair density
        exchange-correlation functional, in the atomic-orbital basis.
        In PDFT this functional is always spin-symmetric
    '''
    if rho.ndim == 2:
        rho = np.expand_dims (rho, 1)
        Pi = np.expand_dims (Pi, 0)

    w = weight[None,:]
    if isinstance (ao, np.ndarray) and ao.ndim == 3:
        ao = [ao, ao]
    elif len (ao) != 2:
        raise NotImplementedError ("uninterpretable aos!")
    if kern is None: 
        kern = otfnal.get_dEot_drho (rho, Pi, **kwargs) 
    else:
        kern = kern.copy () 
    nderiv = kern.shape[0]
    kern *= weight[None,:]

    # Zeroth and first derivatives
    veff = np.tensordot (kern[0,:,None] * ao[0][0], ao[1][0], axes=(0,0))
    if nderiv > 1:
        veff += np.tensordot ((kern[1:4,:,None] * ao[0][1:4]).sum (0), ao[1][0], axes=(0,0))
        veff += np.tensordot (ao[0][0], (kern[1:4,:,None] * ao[1][1:4]).sum (0), axes=(0,0))

    rho = np.squeeze (rho)
    Pi = np.squeeze (Pi)

    return veff

def get_veff_2body (otfnal, rho, Pi, ao, weight, kern=None, **kwargs):
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

    Kwargs:
        kern : ndarray of shape (*,ngrids)
            the derivative of the on-top potential with respect to pair density (vot)
            If not provided, it is calculated.

    Returns : eri-like ndarray
        The two-body effective potential corresponding to this on-top pair density
        exchange-correlation functional or elements thereof, in the provided basis.
    '''
    if rho.ndim == 2:
        rho = np.expand_dims (rho, 1)
        Pi = np.expand_dims (Pi, 0)

    if isinstance (ao, np.ndarray) and ao.ndim == 3:
        ao = [ao,ao,ao,ao]
    elif len (ao) != 4:
        raise NotImplementedError ('fancy orbital subsets and fast evaluation in get_veff_2body')
    #elif isinstance (ao, np.ndarray) and ao.ndim == 4:
    #    ao = [a for a in ao]

    if kern is None:
        kern = otfnal.get_dEot_dPi (rho, Pi, **kwargs) 
    else:
        kern = kern.copy ()
    kern *= weight[None,:]
    nderiv, ngrid = kern.shape

    # Flatten deriv and grid so I can tensordot it all at once
    # Index symmetry can be built into _contract_ao1_ao2
    vao2 = _contract_ao1_ao2 (ao[0], ao[1], nderiv, vot=kern)
    vao2 = vao2.reshape (nderiv * ngrid, *vao2.shape[2:])
    ao2 = _contract_ao1_ao2 (ao[2], ao[3], nderiv)
    ao2 = ao2.reshape (nderiv * ngrid, *ao2.shape[2:])
    veff = np.tensordot (vao2, ao2, axes=(0,0))

    '''
    # Zeroth derivative
    ao1 = ao[0][:nderiv,:,:,None] * ao[1][0][None,:,None,:]
    ao2 = ao[2][:nderiv,:,:,None] * ao[3][0][None,:,None,:]
    veff = np.tensordot (kern[0,:,None,None] * ao1[0], ao2[0], axes=(0,0))
    #veff = np.einsum ('g,gp,gq,gr,gs->pqrs',kern[0],ao[0][0],ao[1][0],ao[2][0],ao[3][0])

    # First derivatives
    if nderiv > 1:
        # Index 12
        ao1[1:4] += ao[0][0][None,:,:,None] * ao[1][1:4,:,None,:]
        veff += np.tensordot ((kern[1:4,:,None] * ao1[1:4]).sum (0), ao2[0], axes=(0,0))
        # Index 34
        ao2[1:4] += ao[2][0][None,:,:,None] * ao[3][1:4,:,None,:]
        veff += np.tensordot (ao1[0], (kern[1:4,:,None] * ao2[1:4]).sum (0), axes=(0,0))
    '''
    rho = np.squeeze (rho)
    Pi = np.squeeze (Pi)

    return veff 

def _contract_ao1_ao2 (ao1, ao2, nderiv, vot=None):
    # The vectorization I have in ao1 here seems to make the calculation SLOWER???
    # For the sake of vectorizing gradient calculations I am treating ao1 specially: it may
    # have more than three dimensions
    # AO1 dimensions: x, deriv, grid, p
    # AO2 dimensions: deriv, grid, q
    # Vot dimensions: deriv, grid
    ao1 = np.expand_dims (ao1, -1)
    ao2 = np.expand_dims (ao2, -2)
    if ao1.ndim == 5:
        ao1 = ao1.transpose (1,2,3,4,0)
        ao2 = np.expand_dims (ao2, -1)
    if vot is not None:
        while vot.ndim < ao1.ndim: vot = np.expand_dims (vot, -1)
    # Now the dimensions of everything are deriv, grid, p, q, x
    prod = ao1[:nderiv] * ao2[0]
    if nderiv > 1:
        prod[1:4] += ao1[0] * ao2[1:4] # Product rule
    if vot is not None:
        ao1 = prod
        ao2 = None
        prod = vot[:nderiv] * ao1[0] # Chain rule: still needs deriv
        if nderiv > 1:
            # Chain rule: deriv accounted for already
            prod[0] += (vot[1:4] * ao1[1:4]).sum (0)
    # OK, now prod has dimensions deriv, grid, p, q, x, and x might not be there
    # If x is there I want it to be before p and q but not before deriv and grid
    # Maybe it makes more sense to fix this later but for now I'm doing it here
    if prod.ndim == 5:
        prod = prod.transpose (0,1,4,2,3)
    return prod 

def _contract_vot_rho (vot, rho, add_vrho=None):
    ''' Make a jk-like vrho from vot and a density. k = j so it's just vot * vrho / 2,
        but the product rule needs to be followed '''
    nderiv = vot.shape[0]
    vrho = vot * rho[0]
    if nderiv > 1:
        vrho[0] += (vot[1:4] * rho[1:4]).sum (0)
    vrho /= 2
    # vot involves lower derivatives than vhro in original translation
    # make sure vot * rho gets added to only the proper component(s)
    if add_vrho is not None:
        add_vrho[:nderiv] += vrho
        vrho = add_vrho
    return vrho



