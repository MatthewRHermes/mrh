import numpy as np

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

    Returns : ndarray of shape (2,nao,nao)
        The 1-body effective potential corresponding to this on-top pair density
        exchange-correlation functional, in the atomic-orbital basis.
    '''

    nderiv = Pi.shape[0]
    kern = otfnal.get_dEot_drho (rho, Pi, **kwargs)
    # Zeroth derivative
    veff = (kern[:,0,:,None,None] * ao[None,0,:,None] * ao[None,0,None,:]).sum (1)
    # First derivatives
    for ideriv in range (1,min(nderiv,4)):
        veff_ideriv += (kern[:,ideriv,:,None,None] * ao[None,ideriv,:,None] * ao[None,0,None,:]).sum (1)
        veff += veff_ideriv + veff_ideriv.transpose (0,2,1)
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
    nderiv = Pi.shape[0]
    kern = otfnal.get_dEot_dPi (rho, Pi, **kwargs)
    # Zeroth derivative
    veff = (kern[0,:,None,None,None,None] * ao[0][0,:,:,None,None,None]
        * ao[1][0,:,None,:,None,None] * ao[2][0,:,None,None,:,None]
        * ao[3][0,:,None,None,None,:]).sum (0)
    # First derivatives
    for ideriv in range (1,min(nderiv,4)):
        veff += (kern[ideriv,:,None,None,None,None]
            * ao[0][ideriv,:,:,None,None,None]
            * ao[1][0,     :,None,:,None,None]
            * ao[2][0,     :,None,None,:,None]
            * ao[3][0,     :,None,None,None,:]).sum (0)
        veff += (kern[ideriv,:,None,None,None,None]
            * ao[0][0,     :,:,None,None,None]
            * ao[1][ideriv,:,None,:,None,None]
            * ao[2][0,     :,None,None,:,None]
            * ao[3][0,     :,None,None,None,:]).sum (0)
        veff += (kern[ideriv,:,None,None,None,None]
            * ao[0][0,     :,:,None,None,None]
            * ao[1][0,     :,None,:,None,None]
            * ao[2][ideriv,:,None,None,:,None]
            * ao[3][0,     :,None,None,None,:]).sum (0)
        veff += (kern[ideriv,:,None,None,None,None]
            * ao[0][0,     :,:,None,None,None]
            * ao[1][0,     :,None,:,None,None]
            * ao[2][0,     :,None,None,:,None]
            * ao[3][ideriv,:,None,None,None,:]).sum (0)
    return veff



