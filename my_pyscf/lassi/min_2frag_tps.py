from pyscf import lib
import numpy as np
from scipy import linalg
from mrh.my_pyscf.lassi.citools import get_lroots

def hess_ss (ham_pq, si):
    hs = np.dot (ham_pq, si)
    e = np.dot (si.conj (), hs)
    hess = ham_pq - (np.multiply.outer (si.conj (), si) * e)
    hs -= e*si
    hess -= np.multiply.outer (si.conj (), hs)
    hess -= np.multiply.outer (hs.conj (), si)
    hess += hess.conj ().T
    return hess

def hess_us (ham_pq, si, lroots, nroots):
    nstates = ham_pq.shape[1]
    p = np.prod (lroots)
    hs = np.dot (ham_pq, si)
    si_p = si[:p].reshape (lroots[1], lroots[0])
    hs_p = hs[:p].reshape (lroots[1], lroots[0])
    h_px = ham_pq[:p].reshape (lroots[1], lroots[0], nstates)
    fu = np.dot (si_p.conj (), hs_p.T)
    fv = np.dot (si_p.conj ().T, hs_p)
    hess_us = np.multiply.outer (-(fu - fu.T), si + si.conj ())
    hess_vs = np.multiply.outer (-(fv - fv.T), si + si.conj ())
    dh = np.dot (si_p, h_px)
    hess_us += dh - dh.transpose (1,0,2)
    dh = np.dot (si_p.T, h_px.transpose (1,0,2))
    hess_vs += dh - dh.transpose (1,0,2)
    idx_u = np.repeat (list (range (lroots[1])), lroots[0])
    idx_v = np.tile (list (range (lroots[0])), lroots[1])
    for i in range (nroots):
        delta_u = np.where (idx_u==i)[0]
        hess_us[i,:,delta_u] += hs_p[:,idx_v[delta_u]]
        hess_us[:,i,delta_u] -= hs_p[:,idx_v[delta_u]]
    hess_us = hess_us[nroots:,:nroots,:].reshape (-1,nstates)
    for i in range (nroots):
        delta_v = np.where (idx_v==i)[0]
        hess_vs[i,:,delta_v] += hs_p[idx_u[delta_v],:].T
        hess_vs[:,i,delta_v] -= hs_p[idx_u[delta_v],:].T
    hess_vs = hess_vs[nroots:,:nroots,:].reshape (-1,nstates)
    hess = np.append (hess_us, hess_vs, axis=0)
    return hess + hess.conj ()

def hess_uu (ham_pq, si, lroots, nroots):
    p = np.prod (lroots)
    hs = np.dot (ham_pq, si)[:p].reshape (lroots[1],lroots[0])
    h_pp = ham_pq[:p,:p].reshape (lroots[1],lroots[0],lroots[1],lroots[0])
    si_p = si[:p].reshape (lroots[1],lroots[0])
    hess = lib.einsum ('im,jmln,kn->ijkl', si.conj (), h_pp, si)
    fu = np.dot (si_p.conj (), hs.T)
    for i in range (lroots[1]):
        hess[:,i,i,:] += fu
    hess -= hess.transpose (1,0,2,3)
    hess -= hess.transpose (0,1,3,2)
    nel = (lroots[1]-nroots)*nroots
    hess = hess[nroots:,:nroots,nroots:,:nroots].reshape (nel, nel)
    return hess + hess.conj ()

def hess_vv (ham_pq, si, lroots, nroots):
    p = np.prod (lroots)
    hs = np.dot (ham_pq, si)[:p].reshape (lroots[1],lroots[0])
    h_pp = ham_pq[:p,:p].reshape (lroots[1],lroots[0],lroots[1],lroots[0])
    si_p = si[:p].reshape (lroots[1],lroots[0])
    hess = lib.einsum ('mi,mjnl,nk->ijkl', si.conj (), h_pp, si)
    fv = np.dot (si_p.conj ().T, hs)
    for i in range (lroots[1]):
        hess[:,i,i,:] += fv
    hess -= hess.transpose (1,0,2,3)
    hess -= hess.transpose (0,1,3,2)
    nel = (lroots[0]-nroots)*nroots
    hess = hess[nroots:,:nroots,nroots:,:nroots].reshape (nel, nel)
    return hess + hess.conj ()

def hess_uv (ham_pq, si, lroots, nroots):
    p = np.prod (lroots)
    hs = np.dot (ham_pq, si)[:p].reshape (lroots[1],lroots[0])
    h_pp = ham_pq[:p,:p].reshape (lroots[1],lroots[0],lroots[1],lroots[0])
    si_p = si[:p].reshape (lroots[1],lroots[0])
    hess = lib.einsum ('im,jmnl,nk->ijkl', si.conj (), h_pp, si)
    hess += np.multiply.outer (si.conj (), hs).transpose (0,2,1,3)
    hess -= hess.transpose (1,0,2,3)
    hess -= hess.transpose (0,1,3,2)
    nelu = (lroots[1]-nroots)*nroots
    nelv = (lroots[0]-nroots)*nroots
    hess = hess[nroots:,:nroots,nroots:,:nroots].reshape (nelu, nelv)
    return hess + hess.conj ()

def hess (ham_pq, si, lroots, nroots):
    hess_uu = hess_uu (ham_pq, si, lroots, nroots)
    hess_uv = hess_uv (ham_pq, si, lroots, nroots)
    hess_vv = hess_vv (ham_pq, si, lroots, nroots)
    hess_uu = np.append ([np.append (hess_uu, hess_uv, axis=1),
                          np.append (hess_uv.T, hess_vv, axis=1)],
                         axis=0)
    hess_us = hess_us (ham_pq, si, lroots, nroots)
    hess_ss = hess_ss (ham_pq, si)
    hess = np.append ([np.append (hess_uu, hess_us, axis=1),
                       np.append (hess_us.T, hess_ss, axis=1)],
                      axis=0)
    return hess

def grad (ham_pq, si, lroots, nroots):
    p = np.prod (lroots)
    hs = np.dot (ham_pq, si)
    e = np.dot (si.conj (), hs)
    grad_s = hs - e*si
    hs = hs[:p].reshape (lroots[1], lroots[0])
    si = si[:p].reshape (lroots[1], lroots[0])
    fu = np.dot (si.conj (), hs.T)
    grad_u = fu - fu.T
    grad_u = grad_u[nroots:,:nroots].ravel ()
    fv = np.dot (si.conj ().T, hs)
    grad_v = fv - fv.T
    grad_v = grad_v[nroots:,:nroots].ravel ()
    return np.concatenate ([grad_u, grad_v, grad_s])

def quadratic_step (ham_pq, ci1, si_p, si_q):
    nroots = len (si_p)
    lroots = get_lroots (ci1)
    si = np.zeros (lroots[::-1])
    si[:nroots,:nroots] = np.diag (si_p)
    si = np.append (si.ravel (), si_q)
    x = linalg.solve (hess (ham_pq, si, lroots, nroots),
                      -grad (ham_pq, si, lroots, nroots))
    nelu = (lroots[1]-nroots)*nroots
    nelv = (lroots[0]-nroots)*nroots
    kappa_u = np.zeros ((lroots[1],lroots[1]), dtype=x.dtype)
    kappa_u[nroots:,:nroots] = x[:nelu].reshape (
            lroots[1]-nroots, nroots)
    kappa_u -= kappa_u.T
    u = linalg.expm (kappa_u)[:,:nroots]
    kappa_v = np.zeros ((lroots[0],lroots[0]), dtype=x.dtype)
    kappa_u[nroots:,:nroots] = x[:nelv].reshape (
            lroots[0]-nroots, nroots)
    kappa_v -= kappa_v.T
    vh = linalg.expm (kappa_v)[:,:nroots].conj ().T
    return u, vh


    



