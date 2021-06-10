import numpy as np
from scipy import linalg
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf.grad import mcpdft as mcpdft_grad
from pyscf import lib
from pyscf.mcscf import mc1step

def make_rdm12_heff_offdiag (mc, ci, si_bra, si_ket): 
    ncas, nelecas = mc.ncas, mc.nelecas
    nroots = len (ci)
    ci_arr = np.asarray (ci)
    ci_bra = np.tensordot (si_bra, ci_arr, axes=1)
    ci_ket = np.tensordot (si_ket, ci_arr, axes=1)
    casdm1, casdm2 = mc.fcisolver.trans_rdm12 (ci_bra, ci_ket, ncas, nelecas)
    ddm1 = np.zeros ((nroots, ncas, ncas), dtype=casdm1.dtype)
    ddm2 = np.zeros ((nroots, ncas, ncas, ncas, ncas), dtype=casdm1.dtype)
    for i in range (nroots):
        ddm1[i,...], ddm2[i,...] = mc.fcisolver.make_rdm12 (ci[i], ncas, nelecas)
    si_diag = si_bra * si_ket
    casdm1 -= np.tensordot (si_diag, ddm1, axes=1)
    casdm2 -= np.tensordot (si_diag, ddm2, axes=1)
    return casdm1, casdm2


def sipdft_heff_response (mc_grad, mo_coeff=None, ci=None,
        si_bra=None, si_ket=None, state=None, ham_si=None, eris=None):
    ''' Compute the orbital and intermediate-state rotation response 
        vector in the context of an SI-PDFT gradient calculation '''
    if mo_coeff is None: mo_coeff = mc_grad.mo_coeff
    if ci is None: ci = mc_grad.ci
    if state is None: state = mc_grad.state
    if si_bra is None: si_bra = mc_grad.si[:,state]
    if si_ket is None: si_ket = mc_grad.si[:,state]
    if ham_si is None: ham_si = mc_grad.ham_si.copy ()
    if eris is None: eris = mc.ao2mo (mo_coeff)
    mc = mc_grad.base

    # Orbital rotation
    casdm1, casdm2 = make_rdm12_heff_offdiag (mc, ci, si_bra, si_ket)
    g_orb = mc1step.gen_g_hop (mc, mo_coeff, 1, casdm1, casdm2, eris)[0]

    # Intermediate state rotation
    ham_si[np.diag_indices (nroots)] = 0.0
    braH = np.dot (si_bra, ham_si)
    Hket = np.dot (ham_si, si_ket)
    g_int  = np.multiply.outer (si_ket, braH)
    g_int += np.multiply.outer (si_bra, Hket)
    g_int -= g_int.T
    g_int = g_int[np.tril_indices (nroots, k=-1)]

    return g_orb, g_int

def get_sarotfns (obj):
    if obj.upper () == 'CMS':
        from mrh.my_pyscf.grad.cmspdft import sarot_response, sarot_grad
    else:
        raise RuntimeError ('SI-PDFT type not supported')
    return sarot_response, sarot_grad

class Gradients (mcpdft_grad.Gradients):

    def __init__(self, mc):
        mcpdft_grad.Gradients.__init__(self, mc)
        r, g = get_sarotfns (self.base.sarot_name)
        self._sarot_response = r
        self._sarot_grad = d

    def sarot_response (self, *args, **kwargs): return self._sarot_response (self, *args, **kwargs)
    def sarot_grad (self, *args, **kwargs): return self._sarot_grad (self, *args, **kwargs)

    def get_wfn_response (self, si_bra=None, si_ket=None, state=None, mo=None, ci=None, **kwargs):
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if state is None: state = mc_grad.state
        if si_bra is None: si_bra = mc_grad.si[:,state]
        if si_ket is None: si_ket = mc_grad.si[:,state]
        si_diag = si_bra * si_ket

        # Diagonal: PDFT component

    def project_Aop (self, Aop, ci, state):
        # Remove intermediate-state rotation entirely
        ci_arr = np.asarray (ci).reshape (self.nroots, -1)
        def my_Aop (x):
            x_ci = x[self.ngorb:].reshape (self.nroots, -1)
            cx = ci_arr.conjugate () @ x_ci.T
            x_ci[:,:] -= np.dot (cx.T, ci_arr)
            x[self.ngorb:] = x_ci.ravel () # unnecessary?
            Ax = Aop (x)
            Ax_ci = Ax_ci[self.ngorb:].reshape (self.nroots, -1)
            cAx = ci_arr.conjugate () @ Ax_ci.T
            Ax_ci[:,:] -= np.dot (cAx.T, ci_arr)
            Ax[self.ngorb:] = Ax_ci.ravel () #unnecessary?
            return Ax
        return my_Aop




