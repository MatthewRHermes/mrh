import numpy as np
from scipy import linalg
from pyscf import lib
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
        vot : (ndarray of shape (*,ngrids), ndarray of shape (*,ngrids)) or None
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
    if rho.ndim == 2: rho = rho[:,None,:]
    if Pi.ndim == 1: Pi = Pi[None,:]
    assert (rho.shape[0] == 2)
    nderiv = rho.shape[1]
    nderiv_Pi = Pi.shape[1]
    if nderiv > 4:
        raise NotImplementedError ("Translation of meta-GGA functionals")
    rho_t = otfnal.get_rho_translated (Pi, rho, weights=weights)
    rho_tot = rho.sum (0)
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
        vxc = list (vrho.T)
        if otfnal.dens_deriv: vxc = vxc + list (vsigma.T)
        vot = otfnal.jT_op (vxc, rho, Pi)
        vot = _unpack_sigma_vector (vot, deriv1=rho_deriv, deriv2=Pi_deriv)
    if dderiv > 1:
        # I should implement this entirely in terms of the gradient norm, since that reduces the
        # number of grid columns from 25 to 9 for t-GGA and from 64 to 25 for ft-GGA (and steps
        # around the need to "unpack" fsigma and frhosigma entirely).
        frho, frhosigma, fsigma = xc_grid[2][:3]
        frho = frho.T
        fxc  = [frho[0],]
        fxc += [frho[1],      frho[2],]
        if otfnal.dens_deriv:
            frhosigma, fsigma = frhosigma.T, fsigma.T
            fxc += [frhosigma[0], frhosigma[3], fsigma[0],]
            fxc += [frhosigma[1], frhosigma[4], fsigma[1], fsigma[3],]
            fxc += [frhosigma[2], frhosigma[5], fsigma[2], fsigma[4], fsigma[5]]
        # First pass: fxc
        fot = jT_f_j (otfnal, fxc, otfnal.jT_op, rho, Pi)
        # Second pass: translation derivatives
        fot[:5] += otfnal.d_jT_op (vxc, rho, Pi)
    if (weights is not None) and otfnal.verbose >= logger.DEBUG and dderiv > 1:
        R = otfnal.get_ratio (Pi, rho_tot/2)[0]
        zeta = otfnal.get_zeta (R, fn_deriv=0)[0]
        idx = zeta == 0
        if np.count_nonzero (idx):
            frho = (frho[0] + 2*frho[1] + frho[2])/4
            lib.logger.debug (otfnal,
                'MC-PDFT density Hessian zeta==0 error element 0: %e',
                linalg.norm (fot[0][idx] - frho[idx]))
            for ix, f in enumerate (fot[1:]):
                lib.logger.debug (otfnal,
                    'MC-PDFT density Hessian zeta==0 error element %d: %e',
                    ix+1, linalg.norm (f[idx]))
        drho = rho_tot / 10**4 
        dPi = Pi / 10**4
        # ~~ omit effect of tLDA/tGGA numerical instability @ R=1 begin ~~~
        drho[:,idx] = dPi[:,idx] = 0.0
        rho1 = rho + (drho/2)[None,:,:]
        Pi1 = Pi+dPi
        R1 = otfnal.get_ratio (Pi1, rho1.sum (0)/2)[0]
        zeta1 = otfnal.get_zeta (R1, fn_deriv=0)[0]
        idx = (zeta1 == 0) & (zeta != 0)
        drho[:,idx] = dPi[:,idx] = 0.0
        # ~~ omit effect of tLDA/tGGA numerical instability @ R=1 end ~~~
        rho1 = rho + (drho/2)[None,:,:]
        Pi1 = Pi+dPi
        rho_t_1 = otfnal.get_rho_translated (Pi1, rho1, weights=weights)
        e1, vxc1 = otfnal._numint.eval_xc (otfnal.otxc, (rho_t_1[0], rho_t_1[1]), spin=1,
            relativity=0, deriv=dderiv, verbose=otfnal.verbose)[:2]
        de = np.dot (weights, (e1*rho_t_1.sum (0))[0] - eot)
        lib.logger.debug (otfnal, 'MC-PDFT quadratic expansion test, e1-e0 = %e', de)
        vr = np.dot (vot[0][0] * drho[0], weights)
        lib.logger.debug (otfnal, 'MC-PDFT quadratic expansion test, vrho.drho = %e', vr)
        vP = np.dot (vot[1][0] * dPi[0], weights)
        lib.logger.debug (otfnal, 'MC-PDFT quadratic expansion test, vPi.dPi = %e', vP)
        Hx_rho = fot[0]*drho[0], fot[1]*drho[0]
        Hx_Pi = fot[1]*dPi[0], fot[2]*dPi[0]
        Hx_sg0 = 0, 0, 0
        vx = vr + vP
        if otfnal.dens_deriv:
            vrp = np.dot ((vot[0][1:4] * drho[1:4]).sum (0), weights)
            lib.logger.debug (otfnal, "MC-PDFT quadratic expansion test, vrho'.drho' = %e", vrp)
            vx += vrp
            dsg0   = 2*(rho_tot[1:4]*drho[1:4]).sum (0)
            Hx_rho = tuple (list (Hx_rho) + [fot[3]*drho[0]])
            Hx_Pi  = tuple (list (Hx_Pi)  + [fot[4]*dPi[0]])
            Hx_sg0 = fot[3]*dsg0, fot[4]*dsg0, fot[5]*dsg0
            if len (vot[1]) > 1:
                vPp = np.dot ((vot[1][1:4] * dPi[1:4]).sum (0), weights)
                lib.logger.debug (otfnal, "MC-PDFT quadratic expansion test, vPi'.dPi' = %e", vPp)
                vx += vPp
        frr = np.dot (Hx_rho[0] * drho[0], weights) / 2
        lib.logger.debug (otfnal, 'MC-PDFT quadratic expansion test, drho.frho2.drho / 2 = %e', frr)
        fPr = np.dot (Hx_rho[1] * dPi[0], weights) / 2
        lib.logger.debug (otfnal, 'MC-PDFT quadratic expansion test, dPi.frho2.drho / 2 = %e', fPr)
        frP = np.dot (Hx_Pi[0] * drho[0], weights) / 2
        lib.logger.debug (otfnal, 'MC-PDFT quadratic expansion test, drho.frho2.dPi / 2 = %e', frP)
        fPP = np.dot (Hx_Pi[1] * dPi[0], weights) / 2
        lib.logger.debug (otfnal, 'MC-PDFT quadratic expansion test, dPi.frho2.dPi / 2 = %e', fPP)
        fxx = frr + frP + fPr + fPP
        if otfnal.dens_deriv:
            fsr = np.dot (Hx_rho[2] * dsg0, weights) / 2
            lib.logger.debug (otfnal, 'MC-PDFT quadratic expansion test, dsg0.frho2.drho / 2 = %e', fsr)
            fsP = np.dot (Hx_Pi[2] * dsg0, weights) / 2
            lib.logger.debug (otfnal, 'MC-PDFT quadratic expansion test, dsg0.frho2.dPi / 2 = %e', fsP)
            frs = np.dot (Hx_sg0[0] * drho[0], weights) / 2
            lib.logger.debug (otfnal, 'MC-PDFT quadratic expansion test, drho.frho2.dsg0 / 2 = %e', frs)
            fPs = np.dot (Hx_sg0[1] * dPi[0], weights) / 2
            lib.logger.debug (otfnal, 'MC-PDFT quadratic expansion test, dPi.frho2.dsg0 / 2 = %e', fPs)
            fss = np.dot (Hx_sg0[2] * dsg0, weights) / 2
            lib.logger.debug (otfnal, 'MC-PDFT quadratic expansion test, dsg0.frho2.dsg0 / 2 = %e', fss)
            fxx += fsr + fsP + frs + fPs + fss
        lib.logger.debug (otfnal, 'MC-PDFT quadratic expansion energy test terms = %e , %e', vx, fxx)
        lib.logger.debug (otfnal, 'MC-PDFT quadratic expansion energy test, linear expansion error = %e', de-vx)
        lib.logger.debug (otfnal, 'MC-PDFT quadratic expansion energy test, quadratic expansion error = %e', de-vx-fxx)
        vxc2 = list (vxc1[0].T)
        if otfnal.dens_deriv: vxc2 = vxc2 + list (vxc1[1].T)
        vot1 = otfnal.jT_op (vxc2, rho1, Pi1)
        vot1 = _unpack_sigma_vector (vot1, deriv1=rho_deriv, deriv2=Pi_deriv)
        Hrho_x_test = Hx_rho[0] + Hx_Pi[0] + Hx_sg0[0]
        Hrho_x_ref = vot1[0][0] - vot[0][0]
        lib.logger.debug (otfnal, 'MC-PDFT quadratic expansion gradient test (h.x)_rho - (v1-v0)_rho = %e - %e -> %e', 
            linalg.norm (weights*Hrho_x_test), linalg.norm (weights*Hrho_x_ref), linalg.norm (weights*(Hrho_x_test-Hrho_x_ref)))
        HPi_x_test = Hx_rho[1] + Hx_Pi[1] + Hx_sg0[1]
        HPi_x_ref = vot1[1] - vot[1]
        lib.logger.debug (otfnal, 'MC-PDFT quadratic expansion gradient test (h.x)_Pi - (v1-v0)_Pi = %e - %e -> %e', 
            linalg.norm (weights*HPi_x_test), linalg.norm (weights*HPi_x_ref), linalg.norm (weights*(HPi_x_test-HPi_x_ref)))
        if otfnal.dens_deriv:
            Hsg0_x_test = 2 * (Hx_rho[2] + Hx_Pi[2] + Hx_sg0[2]) * rho_tot[1:4]
            Hsg0_x_ref = vot1[0][1:4] - vot[0][1:4]
            lib.logger.debug (otfnal, 'MC-PDFT quadratic expansion gradient test (h.x)_sg0 - (v1-v0)_sg0 = %e - %e -> %e', 
                linalg.norm (weights*Hsg0_x_test), linalg.norm (weights*Hsg0_x_ref), linalg.norm (weights*(Hsg0_x_test-Hsg0_x_ref)))
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
        unp1[1:4] = 2*deriv1*packed[2]
        if ncol2 > 1:
            unp1[1:4] += deriv2*packed[3]
            unp2[1:4] = (2*deriv2*packed[4]) + (deriv1*packed[3])
    return unp1, unp2

def contract_fot (otfnal, fot, rho0, Pi0, rho1, Pi1):
    r''' Evaluate the product of a packed lower-triangular matrix
    with perturbed density, pair density, and derivatives.

    Args:
        fot : ndarray of shape (*,ngrids)
            Lower-triangular matrix elements corresponding to the basis
            (rho, Pi, |drho|^2, drho'.dPi, |dPi|) stopping at Pi (3 elements)
            for t-LDA or ft-LDA and |drho|^2 (6 elements) for t-GGA.
        rho0 : ndarray of shape (2,*,ngrids)
            containing spin-density [and derivatives]
            the density at which fot was evaluated
        Pi0 : ndarray with shape (*,ngrids)
            containing on-top pair density [and derivatives]
            the density at which fot was evaluated
        rho1 : ndarray of shape (2,*,ngrids)
            containing spin-density [and derivatives]
            the density contracted with fot
        Pi1 : ndarray with shape (*,ngrids)
            containing on-top pair density [and derivatives]
            the density contracted with fot

    Returns: 
        vot1 : (ndarray of shape (*,ngrids), ndarray of shape (*,ngrids))
            product of fot wrt (density, pair density)
            and their derivatives
    '''    
    if rho0.ndim == 2: rho0 = rho0[:,None,:]
    if Pi0.ndim == 1: Pi0 = Pi0[None,:]
    assert (rho0.shape[0] == 2)
    rho0 = rho0.sum (0)
    if rho1.ndim == 2: rho1 = rho1[:,None,:]
    if Pi1.ndim == 1: Pi1 = Pi1[None,:]
    assert (rho1.shape[0] == 2)
    rho1 = rho1.sum (0)

    vrho1, vPi1 = np.zeros_like (rho1), np.zeros_like (Pi1)
    nel = len (fot)
    nr = int (round (np.sqrt (1 + 8*nel) - 1)) // 2
    ngrids = fot[0].shape[-1]

    vrho1[0] = fot[0]*rho1[0] + fot[1]*Pi1[0]
    vPi1[0] = fot[2]*Pi1[0] + fot[1]*rho1[0]

    if len (fot) > 3:
        srr = 2*(rho0[1:4,:]*rho1[1:4,:]).sum (0)
        vrho1[0] += fot[3]*srr
        vPi1[0]  += fot[4]*srr
        vrr = fot[3]*rho1[0] + fot[4]*Pi1[0] + fot[5]*srr
    if len (fot) > 6:
        srP = ((rho0[1:4,:]*Pi1[1:4,:]).sum (0)
             + (rho1[1:4,:]*Pi0[1:4,:]).sum (0))
        sPP = 2*(Pi0[1:4,:]*Pi1[1:4,:]).sum (0)
        vrho1[0] += fot[6]*srP + fot[10]*sPP
        vPi1[0]  += fot[7]*srP + fot[11]*sPP
        vrr      += fot[8]*srP + fot[12]*sPP
        vrP = (fot[6]*rho1[0]  + fot[7]*Pi1[0]
             + fot[8]*srr  +  fot[9]*srP  + fot[13]*sPP)
        vPP = (fot[10]*rho1[0] + fot[11]*Pi1[0]
             + fot[12]*srr +  fot[13]*srP + fot[14]*sPP)

    if len (fot) > 3:
        vrho1[1:4]  = 2*vrr*rho0[1:4]
    if len (fot) > 6:
        vrho1[1:4] += vrP*Pi0[1:4]
        vPi1[1:4] = 2*vPP*Pi0[1:4] + vrP*rho0[1:4]

    return vrho1, vPi1

def jT_f_j (log, frr, jT_op, *args):
    r''' Apply a jacobian function taking *args to the lower-triangular
    second-derivative array frr'''
    nel = len (frr)
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
    fcr = np.stack ([jT_op ([frr[i] for i in ix_row], *args)
           for ix_row in idx_arr], axis=1)

    # second pass
    nc = fcr.shape[0]
    if log.verbose < logger.DEBUG:
        fcc = np.empty ((nc*(nc+1)//2, ngrids), dtype=fcr.dtype)
        i = 0
        for ix_row, fc_row in enumerate (fcr):
            di = ix_row + 1
            j = i + di
            fcc[i:j] = jT_op (fc_row, *args)[:di]
            i = j
    else:
        fcc = np.empty ((nc,nc,ngrids), dtype=fcr.dtype)
        for fcc_row, fcr_row in zip (fcc, fcr):
            fcc_row[:] = jT_op (fcr_row, *args)
        for i in range (1, nc):
            for j in range (i):
                scale = (fcc[i,j] + fcc[j,i])/2
                scale[scale==0] = 1
                logger.debug (log, 'MC-PDFT jT_f_j symmetry check %d,%d: %e', i, j, 
                    linalg.norm ((fcc[i,j]-fcc[j,i])/scale))

    ltri_ix = np.tril_indices (nc)
    fcc = fcc[ltri_ix]

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
    srz = srP * zeta[1]
    szz = sPP * zeta[1] * zeta[1]
    jTx = np.empty_like (x)
    jTx[0] = 2*x[4]*(zeta[0]*srz+rho*szz) + x[3]*srz
    jTx[1] = 2*x[4]*rho*srz
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
    jTx = np.empty_like (x)
    ri = np.empty_like (x)
    ri[0,:] = 0.0
    idx = rho > 1e-15
    ri[0,idx] = 1.0/rho[idx]
    for i in range (4):
        ri[i+1] = ri[i]*ri[0]

    jTx[0] = (-2*R*x[1]*ri[0]
               + x[3]*(6*R*ri[1]*srr - 8*srP*ri[2])
               + x[4]*(-24*R*R*ri[2]*srr + 80*R*ri[3]*srP 
                        - 64*ri[4]*sPP))
    jTx[1] = (4*x[1]*ri[1] - 8*x[3]*ri[2]*srr
              + x[4]*(32*R*ri[3]*srr - 64*ri[4]*srP))
    jTx[2] = -2*R*x[3]*ri[0] + 4*x[4]*R*R*ri[1]
    jTx[3] = 4*x[3]*ri[1] - 16*x[4]*R*ri[2]
    jTx[4] = 16*x[4]*ri[3]
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
    srP = (rho[1:4,:]*Pi[1:4,:]).sum (0)
    sPP = (Pi[1:4,:]*Pi[1:4,:]).sum (0)
    jTx += _ftGGA_jT_op_R2Pi (x, rho, R, srr, srP, sPP)

    return jTx

def gentLDA_d_jT_op (x, rho, Pi, R, zeta):
    rho = rho[0]
    Pi = Pi[0]
    R = R[0]
    ngrid = rho.shape[-1]
    f = np.zeros ((3,ngrid), dtype=x[0].dtype)

    # ab -> cs
    xm = (x[0] - x[1]) / 2.0

    # Indexing
    idx = rho > 1e-15
    rho = rho[idx]
    Pi = Pi[idx]
    xm = xm[idx]
    R = R[idx]
    zeta = zeta[:,idx]

    # Intermediates
    #R = otfnal.get_ratio (Pi, rho/2)
    #zeta = otfnal.get_zeta (R, fn_deriv=2)[1:]
    xmw = 2*xm/rho
    z1 = xmw * (zeta[1] + 2*R*zeta[2])

    # without further ceremony
    f[0,idx] = R*z1
    f[1,idx] = -2*z1/rho
    f[2,idx] = xmw*8*zeta[2]/rho/rho

    return f

def tGGA_d_jT_op (x, rho, Pi, R, zeta):
    # Generates contributions to the first five elements
    # of the lower-triangular packed Hessian
    ngrid = rho.shape[-1]
    f = np.zeros ((5,ngrid), dtype=x[0].dtype)

    # Indexing
    idx = rho[0] > 1e-15
    rho = rho[0:4,idx]
    Pi = Pi[0:1,idx]
    x = [xi[idx] for xi in x]
    R = R[0,idx]
    zeta = zeta[:,idx]

    # ab -> cs
    xcm = (x[2] - x[4]) / 2.0
    xmm = (x[2] + x[4] - x[3]) / 4.0

    # Intermediates
    sigma = (rho[1:4]*rho[1:4]).sum (0)
    rho = rho[0]
    rho2 = rho*rho
    rho3 = rho2*rho
    rho4 = rho3*rho

    # coefficient of dsigma dz
    xcm += 2*zeta[0]*xmm
    f[3,idx] = -2*xcm*R*zeta[1]/rho
    f[4,idx] = 4*xcm*zeta[1]/rho2
    
    # coefficient of d^2 z
    xcm *= sigma
    f[0,idx] = 2*xcm*R*(3*zeta[1] + 2*R*zeta[2])/rho2
    f[1,idx] = -8*xcm*(zeta[1] + R*zeta[2])/rho3
    f[2,idx] = 16*xcm*zeta[2]/rho4

    # coefficient of dz dz
    xmm *= 8*sigma*zeta[1]*zeta[1]/rho2
    f[0,idx] += xmm*R*R
    f[1,idx] -= 2*xmm*R/rho
    f[2,idx] += 4*xmm/rho2
    
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



