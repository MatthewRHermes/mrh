import numpy as np
from scipy import linalg
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf.grad import mcpdft as mcpdft_grad
from pyscf import lib
from pyscf.lib import logger
from pyscf.mcscf import mc1step
from pyscf.grad import rhf as rhf_grad
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

    # Preconditioner solves the IS problem; hence, get_init_guess rewrite unnecessary
    get_init_guess = sacasscf_grad.Gradients.get_init_guess
    project_Aop = sacasscf_grad.Gradients.project_Aop

    def __init__(self, mc):
        mcpdft_grad.Gradients.__init__(self, mc)
        r, g = get_sarotfns (self.base.sarot_name)
        self._sarot_response = r
        self._sarot_grad = d
        self.nlag += self.nis

    @property
    def nis (self): return self.nroots * (self.nroots - 1) // 2

    def sarot_response (self, Lis, **kwargs): return self._sarot_response (self, Lis, **kwargs)
    def sarot_grad (self, Lis, **kwargs): return self._sarot_grad (self, Lis, **kwargs)

    def kernel (self, state=None, mo=None, ci=None, **kwargs):
        ''' Cache the Hamiltonian and effective Hamiltonian terms, and pass
            around the IS hessian

            eris, veff1, veff2, and d2f should be available to all top-level
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
        veff1 = []
        veff2 = []
        d2f = self.base.sarot_objfn (ci=ci)[2]
        for c in ci:
            v1, v2 = self.base.get_pdft_veff (mo, c, incl_coul=True, paaa_only=True)
            veff1.append (v1)
            veff2.append (v2)
        return mcpdft_grad.Gradients.kernel (state=state, mo=mo, ci=ci, d2f=d2f,
            veff1=veff1, veff2=veff2, **kwargs)

    def pack_uniq_var (self, xorb, xci, xis=None):
        x = sacasscf_grad.Gradients.pack_uniq_var (self, xorb, xci)
        if xis is not None: x = np.append (x, xis)
        return x

    def unpack_uniq_var (self, x):
        ngorb, nci, nroots, nis = self.ngorb, self.nci, self.nroots, self.nis
        x, xis = x[:ngorb+nci], x[ngorb+nci:]
        xorb, xci = sacasscf_grad.Gradients.unpack_uniq_var (x)
        if len (xis)==nis: return xorb, xci, xis
        return xorb, xci

    def get_wfn_response (self, si_bra=None, si_ket=None, state=None, mo=None,
            ci=None, eris=None, veff1=None, veff2=None, **kwargs):
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if state is None: state = mc_grad.state
        if si_bra is None: si_bra = mc_grad.si[:,state]
        if si_ket is None: si_ket = mc_grad.si[:,state]
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

        # Separate IS part (TODO: state-average-mix)
        ci_arr = np.asarray (ci).reshape (nroots, -1)
        gci = g_all[ngorb:].reshape (nroots, -1)
        g_is = np.dot (gci.conj (), ci_arr.T)
        g_all[ngorb:] -= np.dot (g_is, ci_arr.conj ()).ravel ()
        g_is = g_is[np.tril_indices (nroots, k=-1)]

        g_all = np.append (g_all, g_is)
        return g_all

    def get_Aop_Adiag (self, verbose=None, mo=None, ci=None, eris=None,
            level_shift=None, d2f=None, **kwargs):
        if verbose is None: verbose = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if eris is None and self.eris is None:
            eris = self.eris = self.base.ao2mo (mo)
        elif eris is None:
            eris = self.eris
        if d2f is None: d2f = self.base.sarot_objfn (ci=ci)[2]
        fcasscf = self.make_fcasscf_sa ()
        hop, Adiag = newton_casscf.gen_g_hop (fcasscf, mo, ci, eris, verbose)[2:]
        hop = self.project_Aop (hop, ci, 0), Adiag
        ngorb, nci = self.ngorb, self.nci
        # TODO: cacheing sarot_response?
        def Aop (x):
            x_v, x_is = x[:ngorb+nci], x[ngorb+nci:]
            Ax_v = hop (x_v) + self.sarot_response (x_is, mo=mo, ci=ci, eris=eris)
            Ax_is = np.dot (d2f, x_is)
            return np.append (Ax_v, Ax_is)
        return Aop, Adiag

    def get_lagrange_precond (self, Adiag, level_shift=None, ci=None, d2f=None, **kwargs):
        if level_shift is None: level_shift = self.level_shift
        if ci is None: ci = self.base.ci
        if d2f is None: d2f = self.base.sarot_objfn (ci=ci)[2]
        return SIPDFTLagPrec (Adiag=Adiag, level_shift=level_shift, ci=ci, 
            d2f=d2f, grad_method=self)

    def get_ham_response (self, si_bra=None, si_ket=None, state=None, mo=None,
            ci=None, eris=None, veff1=None, veff2=None, **kwargs):
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

    def get_LdotJnuc (self, Lvec, atmlst=None, verbose=None,
            ci=None, eris=None, mf_grad=None, **kwargs):
        ''' Add the IS component '''
        if atmlst is None: atmlst = self.atmlst
        if verbose is None: verbose = self.verbose
        if ci is None: ci = self.base.ci[state]
        if eris is None and self.eris is None:
            eris = self.eris = self.base.ao2mo (mo)
        elif eris is None:
            eris = self.eris

        ngorb, nci, nis = self.ngorb, self.nci, self.nis
        Lvec_v, Lvec_is = Lvec[:ngorb+nci], Lvec[ngorb+nci:]

        # Orbital and CI components
        de_Lv = sacasscf_grad.get_LdotJnuc (Lvec_v, atmlst=atmlst,
            verbose=verbose, ci=ci, eris=eris, mf_grad=mf_grad,
            **kwargs)

        # SI component
        t0 = (logger.process_clock(), logger.perf_counter())
        de_Lis = self.sarot_grad (Lvec_is, atmlst=atmlst, mf_grad=mf_grad,
            eris=eris, mo=mo, ci=ci, **kwargs)
        logger.info (self, 
            '--------------- %s gradient Lagrange IS response ---------------',
            self.base.__class__.__name__)
        if verbose >= logger.INFO: rhf_grad._write(self, self.mol, de_Lis, atmlst)
        logger.info (self,
            '----------------------------------------------------------------')
        t0 = logger.timer (self, '{} gradient Lagrange IS response'.format (
            self.base.__class__.__name__), *t0)
        return de_Lv + de_Lis



class SIPDFTLagPrec (sacasscf_grad.SACASLagPrec):
    ''' Solve IS part exactly, then do everything else the same '''

    def __init__(self, Adiag=None, level_shift=None, ci=None, grad_method=None,
            d2f=d2f, **kwargs):
        sacasscf_grad.SACASLagPrec.__init__(self, Adiag=Adiag,
            level_shift=level_shift, ci=ci, grad_method=grad_method)
        self._init_is (d2f=d2f, **kwargs)

    def _init_d2f (d2f=None, **kwargs): self.d2f=d2f

    def __call__(self, x):
        xorb, xci, xis = self.unpack_uniq_var (x)
        Mxorb = self.orb_prec (xorb)
        Mxci = self.ci_prec (xci)
        Mxis = self.is_prec (xis)
        return self.pack_uniq_var (Mxorb, Mxci, Mxis)

    def is_prec (xis): Mxis = linalg.solve (self.d2f, -xis)

