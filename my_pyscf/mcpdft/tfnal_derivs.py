import numpy as np
from scipy import linalg

def eval_ot (otfnal, rho, Pi, weights=None):
    r''' get the integrand of the on-top xc energy and its functional derivatives wrt rho and Pi 

    Args:
        rho : ndarray of shape (2,*,ngrids)
            containing spin-density [and derivatives]
        Pi : ndarray with shape (*,ngrids)
            containing on-top pair density [and derivatives]

    Kwargs:
        weights : ndarray of shape (ngrids)
            used only for debugging the total number of ``translated''
            electrons in the calculation of rho_t

    Returns: 
        eot : ndarray of shape (ngrids)
            integrand of the on-top exchange-correlation energy
        vot1 : ndarray of shape (*,ngrids)
            functional derivative of Eot wrt density and its derivatives
        vot2 : ndarray of shape (*,ngrids)
            functional derivative of Eot wrt pair density and its derivatives
    '''
    nderiv = rho.shape[1]
    rho_t = otfnal.get_rho_translated (Pi, rho, weights=weights)
    eot, vdens = otfnal._numint.eval_xc (otfnal.otxc, (rho_t[0,:,:], rho_t[1,:,:]), spin=1, relativity=0, deriv=1, verbose=otfnal.verbose)[:2]
    vrho, vsigma = vdens[0], vdens[1]
    vxc = np.zeros_like (rho)
    vxc[:,0,:] = vrho.T
    # I'm guessing about the factors below based on the idea that only one of the two product-rule terms 
    if nderiv > 1:
        vxc[0,1:4,:]  = rho_t[0,1:4] * vsigma[:,0] * 2 # sigma_uu; I'm guessing about the factor based on pyscf.dft.numint._uks_gga_wv0!
        vxc[0,1:4,:] += rho_t[1,1:4] * vsigma[:,1]     # sigma_ud
        vxc[1,1:4,:]  = rho_t[0,1:4] * vsigma[:,1]     # sigma_ud
        vxc[1,1:4,:] += rho_t[1,1:4] * vsigma[:,2] * 2 # sigma_dd

    eot *= rho_t[:,0,:].sum (0)
    vrho = otfnal.get_dEot_drho (rho, Pi, vxc=vxc)
    vot = otfnal.get_dEot_dPi (rho, Pi, vxc=vxc)
    return eot, vrho, vot

def get_bare_vxc (otfnal, rho, Pi, weights=None):
    r''' get the functional derivatives dE_ot/drho_t.
    Wrapper to the existing PySCF routines with a little bit of extra math

    Args:
        rho : ndarray of shape (2,*,ngrids)
            containing spin-density [and derivatives]
        Pi : ndarray with shape (*,ngrids)
            containing on-top pair density [and derivatives]

    Kwargs:
        weights : ndarray of shape (ngrids)
            used only for debugging the total number of ``translated''
            electrons in the calculation of rho_t

    Returns: ndarray of shape (2,*,ngrids)
        The bare vxc
    '''
    nderiv = rho.shape[1]
    rho_t = otfnal.get_rho_translated (Pi, rho, weights=weights)
    vrho, vsigma = otfnal._numint.eval_xc (otfnal.otxc, (rho_t[0,:,:], rho_t[1,:,:]), spin=1, relativity=0, deriv=1, verbose=otfnal.verbose)[1][:2]
    vxc = np.zeros_like (rho)
    vxc[:,0,:] = vrho.T
    # I'm guessing about the factors below based on the idea that only one of the two product-rule terms 
    if nderiv > 1:
        vxc[0,1:4,:]  = rho_t[0,1:4] * vsigma[:,0] * 2 # sigma_uu; I'm guessing about the factor based on pyscf.dft.numint._uks_gga_wv0!
        vxc[0,1:4,:] += rho_t[1,1:4] * vsigma[:,1]     # sigma_ud
        vxc[1,1:4,:]  = rho_t[0,1:4] * vsigma[:,1]     # sigma_ud
        vxc[1,1:4,:] += rho_t[1,1:4] * vsigma[:,2] * 2 # sigma_dd

    return vxc


def get_dEot_drho (otfnal, rho, Pi, Rmax=1, zeta_deriv=False, vxc=None):
    r''' get the functional derivative dE_ot/drho
    For translated functionals, this is based on the chain rule:
    dEot/drho_t * drho_t/drho.
    dEot_drho_t comes from PySCF's existing codes
    rho_t = 1/2 * rho_tot * (1 - zeta)
    zeta = (1-R)**(1/2)
    R = 4*Pi/(rho_tot**2)
    
    drho_t[0,0]/drho[0,0] = drho_t[0,0]/drho[1,0] = 1/2 (1 + zeta + R/zeta)
    drho_t[1,0]/drho[0,0] = drho_t[1,0]/drho[1,0] = 1/2 (1 - zeta - R/zeta)

    Args:
        rho : ndarray of shape (2,*,ngrids)
            containing spin-density [and derivatives]
        Pi : ndarray with shape (*,ngrids)
            containing on-top pair density [and derivatives]

    Kwargs:
        Rmax : float
            For ratios above this value, rho is not translated and therefore
            the effective potential kernel is the same as in standard KS-DFT
        zeta_deriv : logical
            If true, propagate derivatives through the zeta intermediate as in
            ``fully-translated'' PDFT
        vxc : ndarray of shape (2,*,ngrids)
            functional derivative of the on-top xc energy wrt translated densities

    Returns: ndarray of shape (*,ngrids)
        The functional derivative of the on-top pair density exchange-correlation
        energy wrt to total density and its derivatives
        The potential must be spin-symmetric in pair-density functional theory
    '''
    nderiv, ngrid = rho.shape[1:]
    nderiv_zeta = nderiv if zeta_deriv else 1
    if vxc is None:
        vxc = otfnal.get_bare_vxc (rho, Pi)
    rho_tot = rho.sum (0)
    R = otfnal.get_ratio (Pi[0:nderiv_zeta,:], rho_tot[0:nderiv_zeta,:]/2)

    # The first term is just the average of the two spin components of vxc; other terms involve the difference
    vot = vxc.sum (0) / 2
    vdiff = (vxc[0] - vxc[1]) / 2

    # Be careful with this indexing!!
    idx = (rho_tot[0] >= 1e-15) & (Pi[0] >= 1e-15) & (Rmax > R[0])
    zeta = np.empty_like (R[:,idx])
    zeta[0] = np.sqrt (1.0 - R[0,idx])

    # Zeroth and first derivatives both have a term of vdiff * zeta
    vot[:,idx] += vdiff[:,idx] * zeta[None,0,:]    

    # Zeroth derivative has a couple additional terms
    RoZ = R[0,idx] / zeta[0]
    vot[0,idx] += vdiff[0,idx] * RoZ
    if nderiv > 1:
        vot[0,idx] += (vdiff[1:4,idx] * rho_tot[1:4,idx]).sum (0) * RoZ / rho_tot[0,idx]

    return vot

        
def get_dEot_dPi (otfnal, rho, Pi, Rmax=1, zeta_deriv=False, vxc=None):
    r''' get the functional derivative dE_ot/dPi

    For translated functionals, this is based on the chain rule:
    dEot/drho_t * drho_t/drho.
    dEot_drho_t comes from PySCF's existing codes
    rho_t = 1/2 * rho_tot * (1 - zeta)
    zeta = (1-R)**(1/2)
    R = 4*Pi/(rho_tot**2)

    The derivative of Pi doesn't contribute until we do the fully-translated functional

    Args:
        rho : ndarray of shape (2,*,ngrids)
            containing spin-density [and derivatives]
        Pi : ndarray with shape (*,ngrids)
            containing on-top pair density [and derivatives]

    Kwargs:
        Rmax : float
            For ratios above this value, rho is not translated and therefore
            the effective potential kernel is zero
        zeta_deriv : logical
            If true, propagate derivatives through the zeta intermediate as in
            ``fully-translated'' PDFT
        vxc : ndarray of shape (2,*,ngrids)
            functional derivative of the on-top xc energy wrt translated densities

    Returns: ndarray of shape (*,ngrids)
        The functional derivative of the on-top pair density exchange-correlation
        energy wrt to the on-top pair density and its derivatives
    '''
    nderiv, ngrid = Pi.shape
    nderiv_zeta = nderiv if zeta_deriv else 1
    if vxc is None:
        vxc = otfnal.get_bare_vxc (rho, Pi)
    vot = np.zeros ((1,ngrid))
    rho_tot = rho.sum (0)
    R = otfnal.get_ratio (Pi[0:nderiv_zeta,:], rho_tot[0:nderiv_zeta,:]/2)

    # Vot has no term with zero zeta; its terms have a cofactor of vxc_b - vxc_a
    vdiff = vxc[1] - vxc[0]

    # Be careful with this indexing!!
    idx = (rho_tot[0] >= 1e-15) & (Pi[0] >= 1e-15) & (Rmax > R[0])
    zeta = np.empty_like (R[:,idx])
    zeta[0] = np.sqrt (1.0 - R[0,idx])

    # Zeroth derivative of rho
    rhoZinv = np.reciprocal (rho_tot[0,idx] * zeta[0])
    vot[0,idx] += vdiff[0,idx] * rhoZinv

    # First derivative of rho
    if nderiv > 1:
        vot[0,idx] += (vdiff[1:4,idx] * rho_tot[1:4,idx]).sum (0) * rhoZinv / rho_tot[0,idx]

    return vot




