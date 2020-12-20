import numpy as np
from scipy import linalg
from pyscf import lib
from pyscf.lib import logger

def eval_ot (otfnal, rho, Pi, dderiv=1, weights=None):
    r''' get the integrand of the on-top xc energy and its functional derivatives wrt rho and Pi 

    Args:
        rho : ndarray of shape (2,*ngrids)
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
    nderiv_Pi = Pi.shape[1]
    if nderiv > 4:
        raise NotImplementedError ("Translation of meta-GGA functionals")
    rho_t = otfnal.get_rho_translated (Pi, rho, weights=weights)
    rho_tot = rho_t.sum (0)
    rho_deriv = rho_tot[1:4,:] if nderiv > 1 else None
    Pi_deriv = Pi[1:4,:] if nderiv_Pi > 1 else None
    xc_grid = otfnal._numint.eval_xc (otfnal.otxc, (rho_t[0,:,:], rho_t[1,:,:]), spin=1, 
        relativity=0, deriv=dderiv, verbose=otfnal.verbose)[:dderiv+1]
    eot = xc_grid[0] * rho_t[:,0,:].sum (0)
    if (weights is not None) and otfnal.verbose >= logger.DEBUG:
        nelec = rho_t[0,0].dot (weights) + rho_t[1,0].dot (weights)
        logger.debug (otfnal, 'MC-PDFT: Total number of electrons in (this chunk of) the total density = %s', nelec)
        ms = (rho_t[0,0].dot (weights) - rho_t[1,0].dot (weights)) / 2.0
        logger.debug (otfnal, 'MC-PDFT: Total ms = (neleca - nelecb) / 2 in (this chunk of) the translated density = %s', ms)
    vot = fot = None
    if dderiv > 0:
        vrho, vsigma = xc_grid[1][:2]
        vxc = list (vrho.T) + list (vsigma.T)
        vot = otfnal.jT_op (vxc, rho, Pi)
        vot = _unpack_sigma_vector (vot, deriv1=rho_deriv, deriv2=Pi_deriv)
    if dderiv > 1:
        raise NotImplementedError ("Translation of density derivatives of higher order than 1")
        # I should implement this entirely in terms of the gradient norm, since that reduces the
        # number of grid columns from 25 to 9 for t-GGA and from 64 to 25 for ft-GGA (and steps
        # around the need to "unpack" fsigma and frhosigma entirely).
        frho, frhosigma, fsigma = xc_grid[2]
        fxc  = [frho[0],]
        fxc += [frho[1],      frho[2],]
        if otfnal.dens_deriv:
            fxc += [frhosigma[0], frhosigma[3], fsigma[0],]
            fxc += [frhosigma[1], frhosigma[4], fsigma[1], fsigma[3],]
            fxc += [frhosigma[2], frhosigma[5], fsigma[2], fsigma[4], fsigma[5]]
        # First pass: fxc
        fot = jT_f_j (fxc, otfnal.jT_op, rho, Pi)
        # Second pass: translation derivatives
        fot += otfnal.d_jT_op (vxc, rho, Pi)
    return eot, vot, fot

def _unpack_sigma_vector (packed, deriv1=None, deriv2=None):
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
    ncol1 = 1 + 3 * int ((deriv1 is not None) and len (packed) > 2)
    ncol2 = 1 + 3 * int ((deriv2 is not None) and len (packed) > 3)
    ngrid = packed[0].shape[-1] # Don't assume it's an ndarray
    unp1 = np.empty ((ncol1, ngrid), dtype=packed[0].dtype)
    unp2 = np.empty ((ncol2, ngrid), dtype=packed[0].dtype)
    unp1[0] = packed[0]
    unp2[0] = packed[1]
    if ncol1 > 1:
        unp1[1:4] = 2 * deriv1 * packed[2]
        if ncol2 > 1:
            unp1[1:4] += deriv2 * packed[3]
            unp2[1:4] = (2 * deriv2 * packed[4]) + (deriv1 * packed[3])
    return unp1, unp2

def jT_f_j (frr, jT_op, *args):
    r''' Apply a jacobian function taking *args to the lower-triangular
    second-derivative array frr'''
    nel = len (f)
    nr = int (round (np.sqrt (1 + 8*nel) - 1)) // 2
    ngrids = frr[0].shape[-1]

    # yada yada indexing...
    ltri_ix = np.tril_indices (nr)
    idx_arr = np.zeros ((nr, nr), dtype=np.int)
    idx_arr[ltri_ix] = range (nel)
    idx_arr += idx_arr.T
    diag_ix = np.diag_indices (nr)
    idx_arr[diag_ix] = idx_arr[diag_ix] // 2
    
    # first pass
    fcr = np.stack ([jT_op ([f[i] for i in ix_row], *args)
           for ix_row in idx_arr], axis=1)
    
    # second pass
    nc = fcr.shape[0]
    fcc = np.empty ((nc*(nc+1)//2, ngrids), dtype=fcr.dtype)
    i = 0
    for ix_row, fc_row in enumerate (fcr):
        di = ix_row + 1
        j = i + di
        fcc[i:j] = jT_op (fc_row, *args)[:di]
        i = j

    return fcc
    


def gentLDA_jT_op (x, rho, Pi, R, zeta):
    ngrid = rho.shape[-1]
    if R.ndim > 1: R = R[0]

    # ab -> cs coordinate transformation
    xc = (x[0] + x[1]) / 2.0
    xm = (x[0] - x[1]) / 2.0

    # Charge sector has no explicit rho denominator and so does not require indexing
    jTx = np.zeros ((2, ngrid), dtype=x[0].dtype)
    jTx[0] = xc + xm*(zeta[0]-(2*R*zeta[1]))

    # Spin sector has a rho denominator
    idx = (rho[0] > 1e-15) 
    zeta = zeta[1,idx]
    rho = rho[0,idx]
    xm = xm[idx]
    jTx[1,idx] = 4*xm*zeta/rho

    return jTx

def tGGA_jT_op (x, rho, Pi, R, zeta):
    ngrid = rho.shape[-1]
    jTx = np.zeros ((3, ngrid), dtype=x[0].dtype)
    if R.ndim > 1: R = R[0]
   
    # ab -> cs coordinate transformation 
    xcc = (x[2] + x[4] + x[3]) / 4.0
    xcm = (x[2] - x[4]) / 2.0
    xmm = (x[2] + x[4] - x[3]) / 4.0

    # Gradient-gradient sector
    jTx[2] = xcc + xcm*zeta[0] + xmm*zeta[0]*zeta[0]

    # Density-gradient sector
    idx = (rho[0] > 1e-15) 
    sigma_fac = ((rho[1:4].conj ()*rho[1:4]).sum (0)*zeta[1])
    sigma_fac = ((xcm + 2*zeta[0]*xmm)*sigma_fac)[idx]
    rho = rho[0,idx]
    R = R[idx]
    sigma_fac = -2*sigma_fac/rho
    jTx[0,idx] = R*sigma_fac
    jTx[1,idx] = -2*sigma_fac/rho

    return jTx

def _ftGGA_jT_op_m2z (x, rho, zeta, srP, sPP):
    jTx = np.empty_like (x)
    jTx[0] = ((x[3] + 2*x[4]*zeta[0])*srP*zeta[1]
              + 2*rho*x[4]*sPP*zeta[1]*zeta[1])
    jTx[1] = 2*x[3]*rho*srp*zeta[1]
    jTx[2] = 0
    jTx[3] = (x[3] + 2*x[4]*zeta[0])*rho
    jTx[4] = x[4]*rho*rho
    return jTx

def _ftGGA_jT_op_z2R (x, zeta, srP, sPP):
    jTx = np.empty_like (x)
    jTx[0] = 0
    jTx[1] = (x[1]*zeta[1] * x[3]*srP*zeta[2] +
              2*x[4]*sPP*zeta[1]*zeta[2])
    jTx[2] = 0
    jTx[3] = x[3]*zeta[1]
    jTx[4] = x[4]*zeta[1]*zeta[1]
    return jTx

def _ftGGA_jT_op_R2Pi (x, rho, R, srr, srP, sPP):
    if rho.ndim > 1: rho = rho[0]
    if R.ndim > 1: R = R[0]
    ri = np.empty_like (jTx)
    ri[0,:] = 1.0
    idx = rho>1e-15
    ri[0,idx] /= rho[idx]
    for i in range (4):
        ri[i+1] = ri[i]*ri[0]

    jTx = np.empty_like (x)
    jTx[0] = (-2*R*x[1]*ri[0]
               + x[3]*(6*R*ri[1]*srr - 8*sPP*ri[2])
               + x[4]*(-24*R*R*ri[2]*srr + 80*R*ri[3]*srP 
                        - 64*ri[4]*sPP))
    jTx[1] = (4*x[1]*ri[1] - 8*x[3]*ri[2]*srr
              + x[4]*(32*R*ri[3]*srr - 64*ri[4]*srP))
    jTx[2] = -2*x[3]*ri[0] + 4*x[4]*R*R*ri[1]
    jTx[3] = 4*x[3]*ri[1] - 16*R*ri[2]
    jTx[4] = 16*ri[3]
    return jTx

def ftGGA_jT_op (x, rho, Pi, R, zeta):
    ngrid = rho.shape[-1]
    jTx = np.zeros ((5, ngrid), dtype=x[0].dtype)

    # ab -> cs step
    jTx[2] = (x[2] + x[4] + x[3]) / 4.0
    jTx[3] = (x[2] - x[4]) / 2.0
    jTx[4] = (x[2] + x[4] - x[3]) / 4.0
    x = jTx

    # Intermediates
    srr = (rho[1:4,:]*rho[1:4,:]).sum (0)
    srP = (rho[1:4,:]*R[1:4,:]).sum (0)
    sPP = (R[1:4,:]*R[1:4,:]).sum (0)

    # cs -> rho,zeta step
    x = _ftGGA_jT_op_m2z (x, rho[0], zeta, srP, sPP)
    jTx[0]  = x[0]
    jTx[1:] = 0.0

    # rho,zeta -> rho,R step
    x = _ftGGA_jT_op_z2R (x, zeta, srP, sPP)

    # rho,R -> rho,Pi step
    zeta = None
    srP = (rho[1:4,:]*Pi[1:4,:]).sum (0)
    sPP = (Pi[1:4,:]*Pi[1:4,:]).sum (0)
    jTx += _ftGGA_jT_op_R2Pi (x, rho, R, srr, srP, sPP)

    return jTx

def gentLDA_d_jT_op (x, rho, Pi, R, zeta):
    ngrid = rho.shape[-1]
    f = np.zeros ((3,ngrid), dtype=x[0].dtype)
    zeta = zeta[1:]
    if R.ndim > 1: R = R[0]

    # ab -> cs
    xm = (x[0] - x[1]) / 2.0

    # Indexing
    idx = rho > 1e-15
    rho = rho[0:1,idx]
    Pi = Pi[0:1,idx]
    xm = xm[idx]

    # Intermediates
    #R = otfnal.get_ratio (Pi, rho/2)
    #zeta = otfnal.get_zeta (R, fn_deriv=2)[1:]
    zeta[0] += 2*R*zeta[1] # sloppy notation...
    zeta = 4*zeta*xm[None,:]/rho[None,:] # sloppy notation...

    # without further ceremony
    f[0] = R*zeta[0]
    f[1] = -zeta[0]/rho
    f[2] = 4*zeta[1]/rho/rho

    return f

def tGGA_d_jT_op (x, rho, Pi, R, zeta):
    # Generates contributions to the first five elements
    # of the lower-triangular packed Hessian
    ngrid = rho.shape[-1]
    f = np.zeros ((5,ngrid), dtype=x[0].dtype)

    # Indexing
    idx = rho > 1e-15
    rho = rho[0:4,idx]
    Pi = Pi[0:1,idx]
    x = x[:,idx]

    # ab -> cs
    xcm = (x[2] - x[4]) / 2.0
    xmm = (x[2] + x[4] - x[3]) / 4.0

    # Intermediates
    #R = otfnal.get_ratio (Pi, rho/2)
    #zeta = otfnal.get_zeta (R, fn_deriv=2)
    sigma = (rho[1:4]*rho[1:4]).sum (0)
    rho = rho[0]
    R = R[0,:]
    rho2 = rho*rho
    rho3 = rho2*rho
    rho4 = rho3*rho

    # coefficient of dsigma dz
    xcm += 2*zeta[0]*xmm
    f[3] = -4*xcm*R*zeta[1]/rho
    f[4] = 8*xcm*zeta[1]/rho2
    
    # coefficient of d^2 z
    xcm *= sigma
    f[0] = 2*xcm*R*(3*zeta[1] + 2*R*zeta[2])/rho2
    f[1] = -8*xcm*(zeta[1] - 4*R*zeta[2])/rho3
    f[2] = 16*xcm*zeta[2]/rho4

    # coefficient of dz dz
    xmm *= 8*sigma*zeta[1]*zeta[1]/rho2
    f[0] += xmm*R*R
    f[1] -= 2*xmm*R/rho
    f[2] += 4*xmm/rho2
    
    return f

def ftGGA_d_jT_op (x, rho, Pi, R, zeta):
    raise NotImplementedError ("Second density derivatives for fully-translated GGA functionals")
    # Generates contributions to the first five elements,
    # then 6,7, then 10,11
    # of the lower-triangular packed Hessian
    # (I.E., no double gradient derivatives)
    ngrid = rho.shape[-1]
    # rho,rho ; Pi,rho ; Pi,Pi ; s(rho,rho),rho ; s(rho,rho),Pi ([0:5])
    frr = np.zeros ((5,ngrid), dtype=x[0].dtype) 
    # s(Pi,rho),rho ; s(Pi,rho),Pi ([6:8])
    fPr = np.zeros ((2,ngrid), dtype=x[0].dtype)
    # s(Pi,Pi),rho ; s(Pi,Pi),Pi ([10:12])
    fPP = np.zeros ((2,ngrid), dtype=x[0].dtype)
    # This is a bad idea why I am I doing this
    f = [[frr[0], frr[1], frr[3], fPr[0], fPP[0]],
         [frr[1], frr[2], frr[4], fPr[1], fPP[1]],
         [frr[3], frr[4], None,   None,   None],
         [fPr[0], fPr[1], None,   None,   None],
         [fPP[0], fPP[1], None,   None,   None]]



