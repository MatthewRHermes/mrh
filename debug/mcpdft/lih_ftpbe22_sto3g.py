import numpy as np
from pyscf import gto, scf
from pyscf.lib import logger
from pyscf import mcpdft
from pyscf.mcpdft.otfnal import transfnal
from pyscf.mcpdft.tfnal_derivs import _ftGGA_jT_op, _unpack_sigma_vector

mol = gto.M (atom='Li 0 0 0; H 1.2 0 0', basis='sto-3g', verbose=logger.DEBUG2,
             output='lih_ftpbe22_sto3g.log')
mf = scf.RHF (mol).run ()
mc = mcpdft.CASSCF (mf, 'ftPBE', 2, 2, grids_level=1).run ()
mc_grad = mc.nuc_grad_method ().run ()

def num_vrho (rho, Pi, d):
    rhop = rho.copy ()
    rhop[:,0,:] *= 1+d
    rhom = rho.copy ()
    rhom[:,0,:] *= 1-d
    denom = 2*d*rho[:,0,0].sum ()
    eotp = mc.otfnal.eval_ot (rhop, Pi)[0][0]
    eotm = mc.otfnal.eval_ot (rhom, Pi)[0][0]
    return (eotp-eotm)/denom

def num_vPi (rho, Pi, d):
    Pip = Pi.copy ()
    Pip[0,:] *= 1+d
    Pim = Pi.copy ()
    Pim[0,:] *= 1-d
    denom = 2*d*Pi[0,0]
    eotp = mc.otfnal.eval_ot (rho, Pip)[0][0]
    eotm = mc.otfnal.eval_ot (rho, Pim)[0][0]
    return (eotp-eotm)/denom

def debug_jT_op (ot, rho, Pi, vxc):
    R = ot.get_ratio (Pi, rho/2)
    zeta = ot.get_zeta (R[0], fn_deriv=2)
    vot = _ftGGA_jT_op (vxc, rho, Pi, R, zeta, _incl_tGGA=True)
    vot = _unpack_sigma_vector (vot, deriv1=rho[1:4], deriv2=Pi[1:4])
    vrho, vPi = vot[0][0,0], vot[1][0,0]
    return vrho, vPi

for R in (0.1, 0.95, 1, 1.05, 1.16):
    #if R != 1: continue
    print ("R =",R)
    rho = np.ones ((2,4,1)) / 2
    rho[:,0,:] *= np.random.rand (1)[0]
    rho_tot = rho.sum (0)
    Pi = np.ones ((4,1)) * R * 0.25 * rho_tot[0,0] * rho_tot[0,0]
    rho[:,1:4,:] *= 1-(2*np.random.rand (3))[None,:,None]
    Pi[1:4,:] *= 1-(2*np.random.rand (3))[:,None]
    eot, vot, fot = mc.otfnal.eval_ot (rho, Pi)
    vrho = vot[0][0,0]
    vPi = vot[1][0,0]
    # subtract tGGA contributions to debug _ftGGA_jT_op specifically
    rho_t = mc.otfnal.get_rho_translated (Pi, rho)
    xc_grid = mc.otfnal._numint.eval_xc (
        mc.otfnal.otxc, (rho_t[0], rho_t[1]), spin=1, relativity=0, 
        deriv=1)[:2]
    exc = xc_grid[0] * rho_t[:,0,:].sum (0)
    vxcr, vxcs = xc_grid[1][:2]
    vxc = list (vxcr.T) + list (vxcs.T)
    vot_tGGA = transfnal.jT_op (mc.otfnal, vxc, rho, Pi)
    vrho_tGGA, vPi_tGGA = vot_tGGA[:2,0]
    for p in range (20):
        d = 1.0/(2**p)
        vrho1 = num_vrho (rho, Pi, d) 
        denom = vrho1-vrho_tGGA
        vrho_err = vrho-vrho1
        vrho_rerr = 0
        if abs (denom)>0 and abs (vrho_err)>1e-8:
            vrho_rerr = vrho_err/denom
        vPi1 = num_vPi (rho, Pi, d) 
        denom = vPi1-vPi_tGGA
        vPi_err = vPi-vPi1
        vPi_rerr = 0
        if abs (denom)>0 and abs (vPi_err)>1e-8:
            vPi_rerr = vPi_err/denom
        print (p, vrho_err, vrho_rerr, vPi_err, vPi_rerr, rho_tot[0,0]*vrho_err/(-2*R),
               rho_tot[0,0]*rho_tot[0,0]*vPi_err/4)
    rho = rho.sum (0).ravel ()
    Pi = Pi.ravel ()
    srr = rho[1:4].dot (rho[1:4])
    srP = rho[1:4].dot (Pi[1:4])
    sPP = Pi[1:4].dot (Pi[1:4])
    x0 = np.ravel (vxc[2:])
    x1 = np.zeros_like (x0)
    x1[0] = (x0[0] + x0[2] + x0[1]) / 4.0
    x1[1] = (x0[0] - x0[2]) / 2.0
    x1[2] = (x0[0] + x0[2] - x0[1]) / 4.0
        

