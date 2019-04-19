import numpy as np

def get_bare_vxc (otfnal, rho, Pi):
    r''' get the functional derivative dE_ot/drho_t
    Wrapper to the existing PySCF routines with a little bit of extra math

    Args:
        rho : ndarray of shape (2,*,ngrids)
            containing spin-density [and derivatives]
        Pi : ndarray with shape (*,ngrids)
            containing on-top pair density [and derivatives]


    Returns: ndarray of shape (2,*,ngrids)
        The bare vxc
    '''
    nderiv = Pi.shape[0]
    rho_t = otfnal.get_rho_translated (Pi, rho)
    vrho, vsigma = otfnal._numint.eval_xc (otfnal.otxc, (rho_t[0,:,:], rho_t[1,:,:]), spin=1, relativity=0, deriv=1, verbose=otfnal.verbose)[1][:2]
    vxc = np.zeros_like (rho)
    vxc[:,0,:] = vrho.T
    if nderiv > 1:
        vxc[0,1:4,:]  = rho_t[0,1:4] * vsigma[:,0] * 4 # sigma_uu; I'm guessing about the factor based on pyscf.dft.numint._uks_gga_wv0!
        vxc[0,1:4,:] += rho_t[1,1:4] * vsigma[:,1] * 2 # sigma_ud
        vxc[1,1:4,:]  = rho_t[0,1:4] * vsigma[:,1] * 2 # sigma_ud
        vxc[1,1:4,:] += rho_t[1,1:4] * vsigma[:,2] * 4 # sigma_dd

    return vxc


def get_dEot_drho (otfnal, rho, Pi, Rmax=1)
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


    Returns: ndarray of shape (2,*,ngrids)
        The functional derivative of the on-top pair density exchange-correlation
        energy wrt to spin density and its derivatives
        The two spins should be identical unless I do the cumulant decomposition
        (or I misunderstand how vsigma works)
    '''
    nderiv, ngrid = Pi.shape
    vxc_t = otfnal.get_bare_vxc (rho, Pi)
    vxc = np.zeros_like (rho)
    rho_tot = rho.sum (0)
    R = otfnal.get_ratio (Pi[0:nderiv_zeta,:], rho_tot[0:nderiv_zeta,:]/2)

    # Be careful with this indexing!!
    idx = (rho_tot[0] >= 1e-15) & (Pi[0] >= 1e-15) & (Rmax > R[0])
    zeta = np.empty_like (R[:,idx])
    zeta[0] = np.sqrt (1.0 - R[0,idx])

    # If there is no zeta, they're all 1/2
    drt_dr = np.ones ((2,ngrid), dtype=vxc_t.dtype) * 0.5 
    drt_dr[0,idx] *= 1 + zeta + R[0,idx]/zeta
    drt_dr[1,idx] *= 1 - zeta - R[0,idx]/zeta
    
    # I assume this is where the recognition that you don't need to keep the
    # potentials for the two spins separate comes in.  Oh well.
    vxc[0,0] = vxc_t[0,0]*drt_dr[0] + vxc_t[1,0]*drt_dr[1]

    # First derivatives
    for ideriv in range (1, min (4, nderiv)):
        drt_dr[:,:] = 0.5
        w = zeta + ((R[0,idx]/zeta) * (rho_tot[1,idx]/rho_tot[0,idx]))
        drt_dr[0,idx] *= 1 + w
        drt_dr[1,idx] *= 1 - w
        vxc[0,ideriv] = (vxc_t[0,ideriv]*drt_dr[0]) + (vxc_t[1,ideriv]*drt_dr[1])

    vxc[1,:] = vxc[0,:]
    return vxc

        
def get_dEot_dPi (otfnal, rho, Pi, rmax=1):
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

    Returns: ndarray of shape (*,ngrids)
        The functional derivative of the on-top pair density exchange-correlation
        energy wrt to the on-top pair density and its derivatives
    '''
    nderiv, ngrid = Pi.shape
    vxc_t = otfnal.get_bare_vxc (rho, Pi)
    vxc = np.zeros_like (Pi)
    rho_tot = rho.sum (0)
    R = otfnal.get_ratio (Pi[0:nderiv_zeta,:], rho_tot[0:nderiv_zeta,:]/2)

    # Be careful with this indexing!!
    idx = (rho_tot[0] >= 1e-15) & (Pi[0] >= 1e-15) & (Rmax > R[0])
    zeta = np.empty_like (R[:,idx])
    zeta[0] = np.sqrt (1.0 - R[0,idx])

    # Here, if there's no zeta, the derivative is zero
    drt_dr = np.zeros (ngrid, dtype=vxc_t.dtype)
    drt_dr[idx] = -1/rho_tot[0,idx]/zeta

    # Wait a minute, is this identically zero???
    vxc[0] = vxc_t[0,0]*drt_dr - vxc_t[1,0]*drt_dr
    
    # First derivatives
    for ideriv in range (1, min (4, nderiv)):
        w = drt_dr.copy ()
        w[idx] *= rho_tot[ideriv,idx]/rho_tot[0,idx]
        vxc[ideriv] = (vxc_t[0,ideriv]*w) - (vxc_t[1,ideriv]*w)

    return vxc




