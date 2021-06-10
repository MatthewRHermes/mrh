import numpy as np
from scipy import linalg
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf.grad import mcpdft as mcpdft_grad
from pyscf import lib
from pyscf.mcscf import mc1step
from pyscf.grad import casscf as casscf_grad
from pyscf.grad import sacasscf as sacasscf_grad

def make_rdm12_heff_offdiag (mc, ci, si_bra, si_ket): 
    # TODO: state-average-mix generalization
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
    mc = mc_grad.base
    if mo_coeff is None: mo_coeff = mc_grad.mo_coeff
    if ci is None: ci = mc_grad.ci
    if state is None: state = mc_grad.state
    if si_bra is None: si_bra = mc.si[:,state]
    if si_ket is None: si_ket = mc.si[:,state]
    if ham_si is None: ham_si = mc.ham_si.copy ()
    if eris is None: eris = mc.ao2mo (mo_coeff)
    nroots = mc_grad.nroots

    # Orbital rotation
    casdm1, casdm2 = make_rdm12_heff_offdiag (mc, ci, si_bra, si_ket)
    g_orb = mc1step.gen_g_hop (mc, mo_coeff, 1, casdm1, casdm2, eris)[0]

    # Intermediate state rotation (TODO: state-average-mix generalization)
    ham_si[np.diag_indices (nroots)] = 0.0
    braH = np.dot (si_bra, ham_si)
    Hket = np.dot (ham_si, si_ket)
    g_is  = np.multiply.outer (si_ket, braH)
    g_is += np.multiply.outer (si_bra, Hket)
    g_is -= g_is.T
    ci_arr = np.asarray (ci).reshape (nroots, -1)
    gci = np.dot (g_is, ci_arr)

    return np.concatenate (g_orb, gci.ravel ())

def sipdft_heff_HellmanFeynman (mc_grad, atmlst=None, mo_coeff=None, ci=None,
        si_bra=None, si_ket=None, state=None, eris=None, **kwargs):
    mc = mc_grad.base
    if atmlst is None: atmlst = mc_grad.atmlst
    if mo_coeff is None: mo_coeff = mc_grad.mo_coeff
    if ci is None: ci = mc_grad.ci
    if state is None: state = mc_grad.state
    if si_bra is None: si_bra = mc.si[:,state]
    if si_ket is None: si_ket = mc.si[:,state]
    if eris is None: eris = mc.ao2mo (mo_coeff)
    nroots = mc_grad.nroots
    
    casdm1, casdm2 = make_rdm12_heff_offdiag (mc, ci, si_bra, si_ket)
    dm12 = lambda * args: casdm1, casdm2
    fcasscf = mc_grad.make_fcasscf (state=state,
        fcisolver_attr={'make_rdm12' : dm12})
    fcasscf_grad = casscf_grad.Gradients (fcasscf)
    return fcasscf_grad.kernel (mo_coeff=mo_coeff, ci=ci[state], 
        atmlst=atmlst, verbose=mc_grad.verbose)

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

    @property
    def nis (self): return self.nroots * (self.nroots - 1) // 2
    def sarot_response (self, *args, **kwargs): return self._sarot_response (self, *args, **kwargs)
    def sarot_grad (self, *args, **kwargs): return self._sarot_grad (self, *args, **kwargs)

    def solve_Lagrange_IS_part (self, g_all, ci=None):
        # TODO: state-average-mix generalization
        if ci is None: ci = self.base.ci
        nroots, ngorb = self.nroots, self.ngorb
        ci_arr = np.asarray (ci).reshape (nroots, -1)

        idx = np.tril_indices (nroots, k=-1)
        d2f = self.base.sarot_objfn (ci=ci)
        df = np.dot (bvec[ngorb:].reshape (nroots, -1).conj (), ci_arr.T)
        return linalg.solve (d2f, -df[idx])

    def get_wfn_response (self, si_bra=None, si_ket=None, state=None, mo=None,
            ci=None, eris=None, veff1=None, veff2=None, Lis=None, **kwargs):
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if state is None: state = mc_grad.state
        if si_bra is None: si_bra = mc_grad.si[:,state]
        if si_ket is None: si_ket = mc_grad.si[:,state]
        assert (Lis is not None)
        log = lib.logger.new_logger (self, self.verbose)
        si_diag = si_bra * si_ket
        nroots, ngorb, nci = self.nroots, self.ngorb, self.nci
        ptr_is = ngorb + nci

        # Diagonal: PDFT component
        g_all = 0
        for i, (amp, c, v1, v2) in enumerate (zip (si_diag, ci, veff1, veff2)):
            if not amp: continue
            g_all += amp * mcpdft_grad.Gradients.get_wfn_response (state=i,
                mo=mo, ci=ci, veff1=v1, veff2=v2, **kwargs)

        # Off-diagonal: heff component
        g_all += sipdft_heff_response (self, mo_coeff=mo_coeff, ci=ci,
            si_bra=si_bra, si_ket=si_ket, eris=eris)

        # Solve Lagrange IS part
        Lis[:] = self.solve_Lagrange_IS_part (g_all, ci=ci)

        # Effective response including IS part
        g_all += self.sarot_response (Lis, mo=mo, ci=ci, eris=eris)

        # Double-check IS part worked?
        ci_arr = np.asarray (ci).reshape (nroots, -1)
        gci = g_all[ngorb:].reshape (nroots, -1)
        g_is = np.dot (gci.conj (), ci_arr.T)
        assert (np.amax (np.abs (g_is)) < 1e-8), g_is
        return g_all

    def project_Aop (self, Aop, ci, state):
        ''' How do I put the SI constraint derivatives in this? '''
        # Remove intermediate-state rotation entirely
        # TODO: state-average-mix generalization)
        nroots, ngorb = self.nroots, self.ngorb
        ci = np.asarray (ci).reshape (nroots, -1)
        def my_Aop (x0):
            x = x0.copy ()
            cx = np.dot (ci.conjugate (), x[ngorb:].reshape (nroots, -1))
            x[ngorb:] -= np.dot (ci.T, cx).ravel ()
            Ax = Aop (x)
            cAx = ci.conjugate () @ Ax[ngorb:].reshape (nroots, -1)
            Ax[ngorb:] -= np.dot (ci.T, cAx).ravel ()
            return Ax
        return my_Aop

    def kernel (self, state=None, mo=None, ci=None, **kwargs):
        ''' Cache the Hamiltonian and effective Hamiltonian terms, and create a
            box for the intermediate-state terms of the Lagrange vector, to
            keep them separate from the PCG algorithm until the end 

            eris, veff1, veff2, and Lis should be available to all top-level
            functions: get_wfn_response, get_Aop_Adiag, get_ham_repsonse, and
            get_LdotJnuc
        '''
        if state is None: state = self.state
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if isinstance (ci, np.ndarray): ci = [ci] # hack hack hack...
        if state is None:
            raise NotImplementedError ('Gradient of PDFT state-average energy')
        self.state = state # Not the best code hygiene maybe
        nroots = self.nroots
        Lis = np.zeros (nroots * (nroots - 1) // 2, dtype=mo.dtype)
        veff1 = []
        veff2 = []
        for c in ci:
            v1, v2 = self.base.get_pdft_veff (mo, c, incl_coul=True, paaa_only=True)
            veff1.append (v1)
            veff2.append (v2)
        return mcpdft_grad.Gradients.kernel (state=state, mo=mo, ci=ci, Lis=Lis,
            veff1=veff1, veff2=veff2, **kwargs)

    def get_ham_response (self, si_bra=None, si_ket=None, state=None, mo=None,
            ci=None, eris=None, veff1=None, veff2=None, Lis=None, **kwargs):
        ''' write sipdft heff Hellmann-Feynman calculator; sum over diagonal
            PDFT Hellmann-Feynman terms '''
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if state is None: state = mc_grad.state
        if si_bra is None: si_bra = mc_grad.si[:,state]
        if si_ket is None: si_ket = mc_grad.si[:,state]
        si_diag = si_bra * si_ket

        # Diagonal: PDFT component
        de = 0
        for i, (amp, c, v1, v2) in enumerate (zip (si_diag, ci, veff1, veff2)):
            if not amp: continue
            de += amp * mcpdft_grad.Gradients.get_ham_response (state=i,
                mo=mo, ci=ci, veff1=v1, veff2=v2, eris=eris, **kwargs)

        # Off-diagonal: heff component
        de += sipdft_heff_HellmanFeynman (mo_coeff=mo, ci=ci, si_bra=si_bra,
            si_ket=si_ket, eris=eris, state=state, **kwargs)
        
        return de


    def get_LdotJnuc (self, *args, **kwargs):
        ''' add Lis component using the kwarg (not the Lvec arg) and
            sarot_grad function call '''

