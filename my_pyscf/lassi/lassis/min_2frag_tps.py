from pyscf import lib
import numpy as np
from scipy import linalg
from mrh.my_pyscf.lassi.citools import get_lroots

def get_hess_ss (ham_pq, si, lroots, nroots, proj):
    nstates = ham_pq.shape[1]
    p = np.prod (lroots)
    hs = np.dot (ham_pq, si)
    e = np.dot (si.conj (), hs)
    hess = ham_pq - e*np.eye (ham_pq.shape[0])
    hs -= e*si
    hess -= np.multiply.outer (si.conj (), hs)
    hess -= np.multiply.outer (hs.conj (), si)
    hess += hess.conj ().T
    hess = proj.conj ().T @ hess @ proj
    return hess

def get_hess_us (ham_pq, si, lroots, nroots, proj):
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
        delta_u = (idx_u==i)
        hess_us[i,:,:p][:,delta_u] += hs_p[:,idx_v[delta_u]]
        hess_us[:,i,:p][:,delta_u] -= hs_p[:,idx_v[delta_u]]
    hess_us = hess_us[nroots:,:nroots,:].reshape (-1,nstates)
    for i in range (nroots):
        delta_v = (idx_v==i)
        hess_vs[i,:,:p][:,delta_v] += hs_p[idx_u[delta_v],:].T
        hess_vs[:,i,:p][:,delta_v] -= hs_p[idx_u[delta_v],:].T
    hess_vs = hess_vs[nroots:,:nroots,:].reshape (-1,nstates)
    hess = np.append (hess_us, hess_vs, axis=0)
    hess = np.dot (hess, proj)
    return hess + hess.conj ()

def get_hess_uu (ham_pq, si, lroots, nroots):
    p = np.prod (lroots)
    hs = np.dot (ham_pq, si)[:p].reshape (lroots[1],lroots[0])
    h_pp = ham_pq[:p,:p].reshape (lroots[1],lroots[0],lroots[1],lroots[0])
    si = si[:p].reshape (lroots[1],lroots[0])
    hess = lib.einsum ('im,jmln,kn->ijkl', si.conj (), h_pp, si)
    fu = np.dot (si.conj (), hs.T)
    for i in range (lroots[1]):
        hess[:,i,i,:] += fu
    hess -= hess.transpose (1,0,2,3)
    hess -= hess.transpose (0,1,3,2)
    nel = (lroots[1]-nroots)*nroots
    hess = hess[nroots:,:nroots,nroots:,:nroots].reshape (nel, nel)
    return hess + hess.conj ()

def get_hess_vv (ham_pq, si, lroots, nroots):
    p = np.prod (lroots)
    hs = np.dot (ham_pq, si)[:p].reshape (lroots[1],lroots[0])
    h_pp = ham_pq[:p,:p].reshape (lroots[1],lroots[0],lroots[1],lroots[0])
    si = si[:p].reshape (lroots[1],lroots[0])
    hess = lib.einsum ('mi,mjnl,nk->ijkl', si.conj (), h_pp, si)
    fv = np.dot (si.conj ().T, hs)
    for i in range (lroots[0]):
        hess[:,i,i,:] += fv
    hess -= hess.transpose (1,0,2,3)
    hess -= hess.transpose (0,1,3,2)
    nel = (lroots[0]-nroots)*nroots
    hess = hess[nroots:,:nroots,nroots:,:nroots].reshape (nel, nel)
    return hess + hess.conj ()

def get_hess_uv (ham_pq, si, lroots, nroots):
    p = np.prod (lroots)
    hs = np.dot (ham_pq, si)[:p].reshape (lroots[1],lroots[0])
    h_pp = ham_pq[:p,:p].reshape (lroots[1],lroots[0],lroots[1],lroots[0])
    si = si[:p].reshape (lroots[1],lroots[0])
    hess = lib.einsum ('im,jmnl,nk->ijkl', si.conj (), h_pp, si)
    hess += np.multiply.outer (si.conj (), hs).transpose (0,2,1,3)
    hess -= hess.transpose (1,0,2,3)
    hess -= hess.transpose (0,1,3,2)
    nelu = (lroots[1]-nroots)*nroots
    nelv = (lroots[0]-nroots)*nroots
    hess = hess[nroots:,:nroots,nroots:,:nroots].reshape (nelu, nelv)
    return hess + hess.conj ()

def get_hess (ham_pq, si, lroots, nroots, proj):
    huu = get_hess_uu (ham_pq, si, lroots, nroots)
    huv = get_hess_uv (ham_pq, si, lroots, nroots)
    hvv = get_hess_vv (ham_pq, si, lroots, nroots)
    huu = np.append (np.append (huu, huv, axis=1),
                     np.append (huv.T, hvv, axis=1),
                     axis=0)
    hus = get_hess_us (ham_pq, si, lroots, nroots, proj)
    hss = get_hess_ss (ham_pq, si, lroots, nroots, proj)
    hess = np.append (np.append (huu, hus, axis=1),
                      np.append (hus.T, hss, axis=1),
                      axis=0)
    return hess

def get_grad (ham_pq, si, lroots, nroots, proj):
    p = np.prod (lroots)
    hs = np.dot (ham_pq, si)
    e = np.dot (si.conj (), hs)
    grad_s = hs - e*si
    grad_s = np.dot (grad_s, proj)
    hs = hs[:p].reshape (lroots[1], lroots[0])
    si = si[:p].reshape (lroots[1], lroots[0])
    fu = np.dot (si.conj (), hs.T)
    grad_u = fu - fu.T
    grad_u = grad_u[nroots:,:nroots].ravel ()
    fv = np.dot (si.conj ().T, hs)
    grad_v = fv - fv.T
    grad_v = grad_v[nroots:,:nroots].ravel ()
    return np.concatenate ([grad_u, grad_v, grad_s])

def get_proj (si, lroots, nroots):
    nstates = len (si)
    p = np.prod (lroots)
    idx_p = np.zeros (lroots, dtype=bool)
    idx_p[:nroots,:nroots] = True
    idx_q = np.ones (nstates-p, dtype=bool)
    idx = np.append (idx_p.ravel (), idx_q)
    x = np.eye (nstates)
    x = x[:,idx]
    Q, R = linalg.qr (x.conj ().T @ si[:,None])
    proj = x @ Q[:,1:]
    return proj

def quadratic_step (ham_pq, ci1, si_p, si_q):
    nroots = len (si_p)
    lroots = get_lroots (ci1)
    si = np.zeros (lroots[::-1])
    si[:nroots,:nroots] = np.diag (si_p)
    si = np.append (si.ravel (), si_q)
    ham_pq = ham_pq.copy ()
    e = np.dot (si.conj (), np.dot (ham_pq, si))
    ham_pq -= np.eye (ham_pq.shape[0])*e
    proj = get_proj (si, lroots, nroots)
    hess = get_hess (ham_pq, si, lroots, nroots, proj)
    grad = get_grad (ham_pq, si, lroots, nroots, proj)
    x = linalg.solve (hess, -grad)
    nelu = (lroots[1]-nroots)*nroots
    nelv = (lroots[0]-nroots)*nroots
    kappa_u = np.zeros ((lroots[1],lroots[1]), dtype=x.dtype)
    kappa_u[nroots:,:nroots] = x[:nelu].reshape (
            lroots[1]-nroots, nroots)
    kappa_u -= kappa_u.T
    u = linalg.expm (-kappa_u/2)[:,:nroots]
    kappa_v = np.zeros ((lroots[0],lroots[0]), dtype=x.dtype)
    kappa_v[nroots:,:nroots] = x[:nelv].reshape (
            lroots[0]-nroots, nroots)
    kappa_v -= kappa_v.T
    vh = linalg.expm (-kappa_v/2)[:,:nroots].conj ().T
    return u, vh

def idx_down_ham_pq (ham_pq, lroots, nroots):
    p = np.prod (lroots)
    q = ham_pq.shape[0] - p
    h_pr = ham_pq[:p,:].reshape (lroots[1],lroots[0],p+q)
    h_pr = h_pr[:nroots,:nroots,:].reshape (nroots*nroots,p+q)
    ham_pq = np.append (h_pr, ham_pq[p:,:], axis=0)
    h_rp = ham_pq[:,:p].reshape (nroots*nroots+q,lroots[1],lroots[0])
    h_rp = h_rp[:,:nroots,:nroots].reshape (nroots*nroots+q,nroots*nroots)
    ham_pq = np.append (h_rp, ham_pq[:,p:], axis=1)
    return ham_pq

def subspace_eig (ham_pq, lroots, nroots):
    p = np.prod (lroots)
    q = ham_pq.shape[0] - p
    p0 = nroots*nroots
    evals, evecs = linalg.eigh (idx_down_ham_pq (ham_pq, lroots, nroots))
    si_p = np.zeros ((lroots[1],lroots[0],p0+q), dtype=evecs.dtype)
    si_p[:nroots,:nroots,:] = evecs[:p0,:].reshape (nroots,nroots,p0+q)
    si = np.append (si_p.reshape (p,p0+q), evecs[p0:,:], axis=0)
    return evals, si


    



