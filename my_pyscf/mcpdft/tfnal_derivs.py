import numpy as np
from scipy import linalg
from pyscf.lib import logger

def eval_ot (otfnal, rho, Pi, dderiv=1, weights=None):
    r''' get the integrand of the on-top xc energy and its functional derivatives wrt rho and Pi 

    Args:
        rho : ndarray of shape (2,*,ngrids)
            containing spin-density [and derivatives]
        Pi : ndarray with shape (*,ngrids)
            containing on-top pair density [and derivatives]

    Kwargs:
        dderiv : integer
            Order of derivatives to return
        weights : ndarray of shape (ngrids)
            used ONLY for debugging the total number of ``translated''
            electrons in the calculation of rho_t
            Not multiplied into anything!

    Returns: 
        eot : ndarray of shape (ngrids)
            integrand of the on-top exchange-correlation energy
        vot : (array_like (rho), array_like (Pi)) or None
            first functional derivative of Eot wrt (density, pair density)
            and their derivatives
        fot : ndarray of shape (*,ngrids) or None
            second functional derivative of Eot wrt density, pair density,
            and derivatives; first dimension is lower-triangular matrix elements
            corresponding to the basis (rho, Pi, |drho|^2, drho'.dPi, |dPi|)
            stopping at Pi (3 elements) for t-LDA and |drho|^2 (6 elements)
            for t-GGA.
    '''
    if dderiv > 2:
        raise NotImplementedError ("Translation of density derivatives of higher order than 2")
    nderiv = rho.shape[1]
    if nderiv > 4:
        raise NotImplementedError ("Translation of meta-GGA functionals")
    rho_t = otfnal.get_rho_translated (Pi, rho, weights=weights)
    xc_grid = otfnal._numint.eval_xc (otfnal.otxc, (rho_t[0,:,:], rho_t[1,:,:]), spin=1, 
        relativity=0, deriv=dderiv, verbose=otfnal.verbose)[:dderiv+1]
    eot = xc_grid[0] * rho_t[:,0,:].sum (0)
    if (weights is not None) and otfnal.verbose >= logger.DEBUG:
        nelec = rho_t[0,0].dot (weight) + rho_t[1,0].dot (weight)
        logger.debug (self, 'MC-PDFT: Total number of electrons in (this chunk of) the total density = %s', nelec)
        ms = (rho_t[0,0].dot (weight) - rho_t[1,0].dot (weight)) / 2.0
        logger.debug (self, 'MC-PDFT: Total ms = (neleca - nelecb) / 2 in (this chunk of) the translated density = %s', ms)
    vot = fot = None
    if dderiv > 0:
        vrho, vsigma = xc_grid[1][:2]
        vxc = np.zeros_like (rho)
        vxc[:,0,:] = vrho.T
        # For GGAs, libxc differentiates with respect to
        #   sigma[0] = nabla^2 rhoa
        #   sigma[1] = nabla rhoa . nabla rhob
        #   sigma[2] = nabla^2 rhob
        # So we have to multiply the Jacobian to obtain the requested derivatives:
        #   J[0,nabla rhoa] = 2 * nabla rhoa
        #   J[0,nabla rhob] = 0
        #   J[1,nabla rhoa] = nabla rhob
        #   J[1,nabla rhob] = nabla rhoa
        #   J[2,nabla rhoa] = 0
        #   J[2,nabla rhob] = 2 * nabla rhob
        if nderiv > 1:
            vxc[0,1:4,:]  = rho_t[0,1:4] * vsigma[:,0] * 2 
            vxc[0,1:4,:] += rho_t[1,1:4] * vsigma[:,1]     
            vxc[1,1:4,:]  = rho_t[0,1:4] * vsigma[:,1]     
            vxc[1,1:4,:] += rho_t[1,1:4] * vsigma[:,2] * 2 
        vot = otfnal.jTx_op (rho, Pi, vxc)
    if dderiv > 1:
        raise NotImplementedError ("Translation of density derivatives of higher order than 1")
        # I should implement this entirely in terms of the gradient norm, since that reduces the
        # number of grid columns from 25 to 9 for t-GGA and from 64 to 25 for ft-GGA (and steps
        # around the need to "unpack" fsigma and frhosigma entirely).
        frho, frhosigma, fsigma = xc_grid[2]

    return eot, vot, fot

def get_dEot_drho (otfnal, rho, Pi, vxc, Rmax=1, zeta_deriv=False):
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
        vxc : ndarray of shape (2,*,ngrids)
            functional derivative of the on-top xc energy wrt translated densities

    Kwargs:
        Rmax : float
            For ratios above this value, rho is not translated and therefore
            the effective potential kernel is the same as in standard KS-DFT
        zeta_deriv : logical
            If true, propagate derivatives through the zeta intermediate as in
            ``fully-translated'' PDFT

    Returns: ndarray of shape (*,ngrids)
        The functional derivative of the on-top pair density exchange-correlation
        energy wrt to total density and its derivatives
        The potential must be spin-symmetric in pair-density functional theory
    '''
    nderiv, ngrid = rho.shape[1:]
    nderiv_zeta = nderiv if zeta_deriv else 1
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

        
def get_dEot_dPi (otfnal, rho, Pi, vxc, Rmax=1, zeta_deriv=False):
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
        vxc : ndarray of shape (2,*,ngrids)
            functional derivative of the on-top xc energy wrt translated densities

    Kwargs:
        Rmax : float
            For ratios above this value, rho is not translated and therefore
            the effective potential kernel is zero
        zeta_deriv : logical
            If true, propagate derivatives through the zeta intermediate as in
            ``fully-translated'' PDFT

    Returns: ndarray of shape (*,ngrids)
        The functional derivative of the on-top pair density exchange-correlation
        energy wrt to the on-top pair density and its derivatives
    '''
    nderiv, ngrid = Pi.shape
    nderiv_zeta = nderiv if zeta_deriv else 1
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




