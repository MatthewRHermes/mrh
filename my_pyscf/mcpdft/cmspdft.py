import numpy as np
from itertools import product
from scipy import linalg
from pyscf import gto, dft, ao2mo, fci, mcscf, lib
from pyscf.lib import logger, temporary_env
from pyscf.mcscf.addons import StateAverageMCSCFSolver, state_average_mix, state_average_mix_, state_average
from pyscf.fci import direct_spin1
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf.grad.mcpdft import Gradients

def coulomb_tensor (mc, mo_coeff=None, ci=None, h2eff=None, eris=None):
    ''' Compute w_IJKL = (tu|vx) D^IJ_tu D^KL_vx 

    Args:
        mc : mcscf method instance

    Kwargs:
        mo_coeff : ndarray of shape (nao,nmo)
            Contains molecular orbital coefficients
        ci : list of ndarrays of shape (ndeta,ndetb)
            Contains CI vectors
        h2eff : ndarray of shape [ncas,]*4
            Contains active-space ERIs
        eris : mc_ao2mo.ERI object
            Contains active-space ERIs. Ignored if h2eff is passed; if
            h2eff is not passed then it is constructed from eris.ppaa

    Returns:
        w : ndarray of shape [nroots,]*4
    '''
    if mo_coeff is None: mo_coeff=mc.mo_coeff
    if ci is None: ci = mc.ci
    # TODO: state-average mix extension
    ci = np.asarray (ci)
    ncore, ncas, nelecas = mc.ncore, mc.ncas, mc.nelecas
    nroots, nocc = mc.fcisolver.nroots, ncore + ncas
    if h2eff is None:
        if eris is None: h2eff = mc.get_h2eff (mo_coeff=mo_coeff)
        else: h2eff = np.asarray (eris.ppaa[ncore:nocc,ncore:nocc,:,:])
    h2eff = ao2mo.restore (1, h2eff, ncas)   
 
    row, col = np.tril_indices (nroots)
    tdm1 = np.stack (mc.fcisolver.states_trans_rdm12(ci[col], ci[row], ncas,
        nelecas)[0], axis=0)
    
    w = np.tensordot (tdm1, h2eff, axes=2)
    w = np.tensordot (w, tdm1, axes=((1,2),(1,2)))
    return ao2mo.restore (1, w, nroots)

def e_coul (mc, mo_coeff=None, ci=None, h2eff=None, eris=None):
    ''' Compute the sum of active-space Coulomb energies (the diabatizer
        function for CMS-PDFT) and its first and second derivatives

    Args:
        mc : mcscf method instance

    Kwargs:
        mo_coeff : ndarray of shape (nao,nmo)
            Contains molecular orbital coefficients
        ci : list of ndarrays of shape (ndeta,ndetb)
            Contains CI vectors
        h2eff : ndarray of shape [ncas,]*4
            Contains active-space ERIs
        eris : mc_ao2mo.ERI object
            Contains active-space ERIs. Ignored if h2eff is passed; if
            h2eff is not passed then it is constructed from eris.ppaa

    Returns:
        Qaa : float
            sum of Coulomb energies
        dQaa : ndarray of shape npair = nroots*(nroots-1)/2
            first derivatives of J wrt interstate rotation
        d2Qaa : ndarray of shape (npair,npair)
            Hessian of J wrt interstate rotation
        Qaa_update : callable
            Takes a unitary matrix of shape (nroots, nroots) and returns
            Qaa, dQaa, and d2Qaa as above using the stored Coulomb
            tensor intermediate from this function. 
    '''
    nroots = mc.fcisolver.nroots   
 
    w0 = coulomb_tensor (mc, mo_coeff=mo_coeff, ci=ci, h2eff=h2eff,
        eris=eris)
    Qaa0, dQaa0, d2Qaa0 = _e_coul (w0, nroots)
    def Qaa_update (u=1):
        w1 = ao2mo.incore.full (w0, u, compact=False)
        return _e_coul (w1, nroots)
    return Qaa0, dQaa0, d2Qaa0, Qaa_update

def _e_coul (w_IJKL, nroots):
    npair = nroots * (nroots - 1) // 2
    w_IJKK = np.diagonal (w_IJKL, axis1=2, axis2=3)
    w_IKJK = np.diagonal (w_IJKL, axis1=1, axis2=3)
    w_IJJJ = np.diagonal (w_IJKK, axis1=1, axis2=2)
    
    Qaa = np.trace (w_IJJJ) / 2.0

    tril_mask = np.zeros ([nroots,nroots], dtype=np.bool_)
    tril_mask[np.tril_indices (nroots,k=-1)] = True
    dQaa = 2*(w_IJJJ.T-w_IJJJ)[tril_mask] 
    # My sign convention is row idx = source state; col idx = dest
    # state, lower-triangular positive. The Newton iteration is designed
    # with this in mind and breaks if I flip it. However, regardless of
    # sign convention, the unitary operator parameterized this way
    # always comes out with destination on the rows and source on the
    # columns, because that's just what the word "operator" means: 
    # |f> = U|i>. So when I exponentiate later, I transpose. Don't let
    # the fact that that transpose is mathematically the same as
    # flipping this sign confuse you: the sign here is CORRECT.
    
    v_IJ_K = -4*w_IKJK - 2*w_IJKK
    v_IJ_K += (w_IJJJ+w_IJJJ.T)[:,:,None]
    d2Qaa = np.zeros_like (w_IJKL)
    for k in range (nroots):
        d2Qaa[:,k,k,:] = v_IJ_K[:,:,k]
    d2Qaa -= d2Qaa.transpose (0,1,3,2)
    d2Qaa -= d2Qaa.transpose (1,0,2,3)
    tril_mask2 = np.logical_and.outer (tril_mask, tril_mask)
    d2Qaa = d2Qaa[tril_mask2].reshape (npair, npair)

    return Qaa, dQaa, d2Qaa

# TODO: could this serve a purpos in unittests?
def e_coul_o0 (mc,ci):
    # Old implementation
    nroots = mc.fcisolver.nroots
    ncas, ncore = mc.ncas,mc.ncore
    nocc = ncas + ncore
    rows,col = np.tril_indices(nroots,k=-1)
    pairs = len(rows)
    mo_cas = mc.mo_coeff[:,ncore:nocc]
    ci_array = np.array(ci)
    casdm1 = mc.fcisolver.states_make_rdm1 (ci,ncas,mc.nelecas)
    dm1 = np.dot(casdm1,mo_cas.T)
    dm1 = np.dot(mo_cas,dm1).transpose(1,0,2)
    j = mc._scf.get_j (dm=dm1)
    e_coul = (j*dm1).sum((1,2)) / 2

    trans12_tdm1, trans12_tdm2 = mc.fcisolver.states_trans_rdm12(ci_array[col],
        ci_array[rows],ncas,mc.nelecas)
    trans12_tdm1_array = np.array(trans12_tdm1)
    tdm1 = np.dot(trans12_tdm1_array,mo_cas.T)
    tdm1 = np.dot(mo_cas,tdm1).transpose(1,0,2)
    
    rowscol2ind = np.zeros ((nroots, nroots), dtype=np.integer)
    rowscol2ind[(rows,col)] = list (range (pairs)) 
    rowscol2ind += rowscol2ind.T 
    rowscol2ind[np.diag_indices(nroots)] = -1 
    
    def w_klmn(k,l,m,n,dm,tdm):
        d = dm[k] if k==l else tdm[rowscol2ind[k,l]]
        dm1_g = mc._scf.get_j (dm=d)
        d = dm[m] if m==n else tdm[rowscol2ind[m,n]]
        w = (dm1_g*d).sum ((0,1))
        return w

    def v_klmn(k,l,m,n,dm,tdm):
        if l==m:
            v = (w_klmn(k,n,k,k,dm,tdm)-w_klmn(k,n,l,l,dm,tdm)
                +w_klmn(n,k,n,n,dm,tdm)-w_klmn(k,n,m,m,dm,tdm)
                -4*w_klmn(k,l,m,n,dm,tdm))
        else:
            v = 0
        return v

    dg = mc._scf.get_j (dm=tdm1)
    grad1 = (dg*dm1[rows]).sum((1,2))
    grad2 = (dg*dm1[col]).sum((1,2))
    e_grad = np.zeros(pairs)
    e_grad = 2*(grad1 - grad2)
   
    e_hess = np.zeros((pairs,pairs))  
    for (i, (k,l)), (j, (m,n)) in product (enumerate (zip (rows, col)),
            repeat=2):
        e_hess[i,j] = (v_klmn(k,l,m,n,dm1,tdm1)+v_klmn(l,k,n,m,dm1,tdm1)
                      -v_klmn(k,l,n,m,dm1,tdm1)-v_klmn(l,k,m,n,dm1,tdm1))

    return sum (e_coul), e_grad, e_hess

# TODO: docstring? delete??
#def eff_ham (mc,ci):
#
#    nroots = mc.fcisolver.nroots
#    ci_array = np.array(ci)
#
#    E_int = np.asarray ([mcpdft.mcpdft.kernel (mc, ot=mc.otfnal, ci=c)[0]
#        for c in ci])
#
#    h1, h0 = mc.get_h1eff ()
#
#    h2 = mc.get_h2eff ()
#
#    h2eff = direct_spin1.absorb_h1e (h1, h2, mc.ncas, mc.nelecas, 0.5)
#
#    hc_all = [direct_spin1.contract_2e (h2eff, c, mc.ncas, mc.nelecas) for c in ci]
#
#    Ham = np.tensordot (ci, hc_all, axes=((1,2),(1,2))) 
#
#    for i in range(nroots):
#        Ham[i,i]=E_int[i]
#
#    ms_e, ms_vecs = linalg.eigh(Ham)
#
#    return ms_e, ms_vecs

#TODO: delete?
#def si_newton (mc, maxiter,thrs):
#
#    assert (False)
#    log = lib.logger.new_logger (mc, mc.verbose)
#    nroots = mc.fcisolver.nroots 
#    rows,col = np.tril_indices(nroots,k=-1)
#    pairs = len(rows)
#    u = np.identity(nroots)
#    t = np.zeros((nroots,nroots))
#    t_old = np.zeros((nroots,nroots))
#    conv = False
#    ci_array = np.array(mc.ci)
#    ci_old = ci_array
#    e_c_old = e_coul(mc,ci_array)
#
#    for it in range(maxiter):
#        log.info ("****iter {} ***********".format (it))
#
##       Form U
#        U = linalg.expm(t)
#
##       Rotate T
#        ci_rot = np.tensordot(U,ci_old,1)
#
#        e_c, e_g, e_h = e_coul(mc, ci_rot)
#        e_c = np.array(e_c)
#
#        log.info ("e_coul = {} ; e_g = {}; e_h = {}".format (e_c, e_g, e_h))
#        log.info ("e_coul sum = {} ".format (e_c.sum()))
#
#        log.info ("t = {} ".format (t))
#
#        evals, evecs = linalg.eigh (e_h)
#        evecs = np.array(evecs)
#        log.info ("Hessian eigenvalues: {}".format (evals))
#
#        for i in range(pairs):
#            if 1E-09 > evals[i] and evals[i] > -1E-09:
#               log.info ("Hess is singular!")
#            if evals[i] > 0 :
#                neg = False
#                break
#            neg = True
#        log.info ("Hess diag is neg? {}".format (neg))
#
#        if neg == False :
#            for i in range(pairs):
#                if evals[i] > 0 :
#                    evals[i]=-evals[i]
#
#        diag = np.identity(pairs)*evals
#        e_h = np.dot(np.dot(evecs,diag),evecs.T)
#
#        t_add = linalg.solve(e_h,-e_g)
#        t[:] = 0
#        t[np.tril_indices(t.shape[0], k = -1)] = t_add
#
#        t = t - t.T
#
#        t_old = t.copy()
#
##       Reset Old Values
#
#        ci_old=ci_rot
#
#        grad_norm = np.linalg.norm(e_g)
#        log.info ("grad norm = %f", grad_norm)
#
#        if grad_norm < thrs and neg == True:
#                conv = True
#                break
#        e_c_old = e_c
#    if conv:
#        log.note ("CMS-PDFT intermediate state determination CONVERGED")
#    else:
#        log.note (("CMS-PDFT intermediate state determination did not converge"
#                   " after {} cycles").format (it))
#
#
#    e_cms, ecms_vecs = eff_ham (mc, ci_rot)
#    log.info ("CMS-PDFT final state energies: {}".format (e_cms))
#
#    return e_cms, ecms_vecs, list (ci_rot)
