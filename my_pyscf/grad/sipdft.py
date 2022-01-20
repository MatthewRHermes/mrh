import numpy as np
from scipy import linalg
from mrh.util.la import vector_error
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf.grad import mcpdft as mcpdft_grad
from mrh.my_pyscf.fci.csf import CSFFCISolver
from pyscf import lib
from pyscf.lib import logger
from pyscf.fci import direct_spin1
from pyscf.mcscf import mc1step, newton_casscf
from pyscf.grad import rhf as rhf_grad
from pyscf.grad import casscf as casscf_grad
from pyscf.grad import sacasscf as sacasscf_grad
from itertools import product

def make_rdm12_heff_offdiag (mc, ci, si_bra, si_ket): 
    # TODO: state-average-mix generalization
    ncas, nelecas = mc.ncas, mc.nelecas
    nroots = len (ci)
    ci_arr = np.asarray (ci)
    ci_bra = np.tensordot (si_bra, ci_arr, axes=1)
    ci_ket = np.tensordot (si_ket, ci_arr, axes=1)
    casdm1, casdm2 = direct_spin1.trans_rdm12 (ci_bra, ci_ket, ncas, nelecas)
    ddm1 = np.zeros ((nroots, ncas, ncas), dtype=casdm1.dtype)
    ddm2 = np.zeros ((nroots, ncas, ncas, ncas, ncas), dtype=casdm1.dtype)
    for i in range (nroots):
        ddm1[i,...], ddm2[i,...] = direct_spin1.make_rdm12 (ci[i], ncas, nelecas)
    si_diag = si_bra * si_ket
    casdm1 -= np.tensordot (si_diag, ddm1, axes=1)
    casdm2 -= np.tensordot (si_diag, ddm2, axes=1)
    return casdm1, casdm2

def sipdft_heff_response (mc_grad, mo=None, ci=None,
        si_bra=None, si_ket=None, state=None, ham_si=None, 
        e_mcscf=None, eris=None):
    ''' Compute the orbital and intermediate-state rotation response 
        vector in the context of an SI-PDFT gradient calculation '''
    mc = mc_grad.base
    if mo is None: mo = mc_grad.mo_coeff
    if ci is None: ci = mc_grad.ci
    if state is None: state = mc_grad.state
    if si_bra is None: si_bra = mc.si[:,state]
    if si_ket is None: si_ket = mc.si[:,state]
    if ham_si is None: ham_si = mc.ham_si
    if e_mcscf is None: e_mcscf = mc.e_mcscf
    if eris is None: eris = mc.ao2mo (mo)
    nroots, ncore = mc_grad.nroots, mc.ncore
    moH = mo.conj ().T

    # Orbital rotation (no all-core DM terms allowed!)
    # (Factor of 2 is convention difference between mc1step and newton_casscf)
    casdm1, casdm2 = make_rdm12_heff_offdiag (mc, ci, si_bra, si_ket)
    casdm1 = 0.5 * (casdm1 + casdm1.T)
    casdm2 = 0.5 * (casdm2 + casdm2.transpose (1,0,3,2))
    vnocore = eris.vhf_c.copy ()
    vnocore[:,:ncore] = -moH @ mc.get_hcore () @ mo[:,:ncore]
    with lib.temporary_env (eris, vhf_c=vnocore):
        g_orb = 2 * mc1step.gen_g_hop (mc, mo, 1, casdm1, casdm2, eris)[0]
    g_orb = mc.unpack_uniq_var (g_orb)

    # Intermediate state rotation (TODO: state-average-mix generalization)
    ham_is = ham_si.copy ()
    ham_is[np.diag_indices (nroots)] = e_mcscf
    braH = np.dot (si_bra, ham_is)
    Hket = np.dot (ham_is, si_ket)
    si2 = si_bra * si_ket
    g_is  = np.multiply.outer (si_ket, braH)
    g_is += np.multiply.outer (si_bra, Hket)
    g_is -= 2 * si2[:,None] * ham_is
    g_is -= g_is.T
    g_is = g_is[np.tril_indices (nroots, k=-1)]

    return g_orb, g_is

def sipdft_heff_HellmanFeynman (mc_grad, atmlst=None, mo=None, ci=None, si=None,
        si_bra=None, si_ket=None, state=None, eris=None, mf_grad=None,
        verbose=None, **kwargs):
    mc = mc_grad.base
    if atmlst is None: atmlst = mc_grad.atmlst
    if mo is None: mo = mc.mo_coeff
    if ci is None: ci = mc.ci
    if si is None: si = getattr (mc, 'si', None)
    if state is None: state = mc_grad.state
    if si_bra is None: si_bra = si[:,state]
    if si_ket is None: si_ket = si[:,state]
    if eris is None: eris = mc.ao2mo (mo)
    if mf_grad is None: mf_grad = mc._scf.nuc_grad_method ()
    if verbose is None: verbose = mc_grad.verbose
    ncore, nroots = mc.ncore, mc_grad.nroots
    log = logger.new_logger (mc_grad, verbose)
    ci0 = np.zeros_like (ci[0])

    # CASSCF grad with effective RDMs
    t0 = (logger.process_clock (), logger.perf_counter ())    
    casdm1, casdm2 = make_rdm12_heff_offdiag (mc, ci, si_bra, si_ket)
    casdm1 = 0.5 * (casdm1 + casdm1.T)
    casdm2 = 0.5 * (casdm2 + casdm2.transpose (1,0,3,2))
    dm12 = lambda * args: (casdm1, casdm2)
    fcasscf = mc_grad.make_fcasscf (state=state,
        fcisolver_attr={'make_rdm12' : dm12})
    # TODO: DFeri functionality
    # Perhaps by patching fcasscf.nuc_grad_method?
    fcasscf_grad = fcasscf.nuc_grad_method ()
    #fcasscf_grad = casscf_grad.Gradients (fcasscf)
    de = fcasscf_grad.kernel (mo_coeff=mo, ci=ci0, atmlst=atmlst, verbose=0)

    # subtract nuc-nuc and core-core (patching out simplified gfock terms)
    moH = mo.conj ().T
    f0 = (moH @ mc.get_hcore () @ mo) + eris.vhf_c
    mo_energy = f0.diagonal ().copy ()
    mo_occ = np.zeros_like (mo_energy)
    mo_occ[:ncore] = 2.0
    f0 *= mo_occ[None,:]
    dme0 = lambda * args: mo @ ((f0+f0.T)*.5) @ moH
    with lib.temporary_env (mf_grad, make_rdm1e=dme0, verbose=0):
     with lib.temporary_env (mf_grad.base, mo_coeff=mo, mo_occ=mo_occ):
        # Second level there should become unnecessary in future, if anyone
        # ever gets around to cleaning up pyscf.df.grad.rhf & pyscf.grad.rhf
        dde = mf_grad.kernel (mo_coeff=mo, mo_energy=mo_energy, mo_occ=mo_occ,
            atmlst=atmlst)
    de -= dde
    log.debug ('SI-PDFT gradient off-diagonal H-F terms:\n{}'.format (de))
    log.timer ('SI-PDFT gradient off-diagonal H-F terms', *t0)
    return de

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
        self._sarot_grad = g
        self.nlag += self.nis

    @property
    def nis (self): return self.nroots * (self.nroots - 1) // 2

    def sarot_response (self, Lis, **kwargs): return self._sarot_response (self, Lis, **kwargs)
    def sarot_grad (self, Lis, **kwargs): return self._sarot_grad (self, Lis, **kwargs)

    def kernel (self, state=None, mo=None, ci=None, si=None, _freeze_is=False, 
            **kwargs):
        ''' Cache the Hamiltonian and effective Hamiltonian terms, and pass
            around the IS hessian

            eris, veff1, veff2, and d2f should be available to all top-level
            functions: get_wfn_response, get_Aop_Adiag, get_ham_repsonse, and
            get_LdotJnuc
 
            freeze_is == True sets the is component of the response to zero
            for debugging purposes
        '''
        if state is None: state = self.state
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if si is None: si = self.base.si
        if isinstance (ci, np.ndarray): ci = [ci] # hack hack hack...
        if state is None:
            raise NotImplementedError ('Gradient of PDFT state-average energy')
        self.state = state # Not the best code hygiene maybe
        nroots = self.nroots
        veff1 = []
        veff2 = []
        d2f = self.base.sarot_objfn (ci=ci)[2]
        for ix in range (nroots):
            v1, v2 = self.base.get_pdft_veff (mo, ci, incl_coul=True,
                paaa_only=True, state=ix)
            veff1.append (v1)
            veff2.append (v2)
        return mcpdft_grad.Gradients.kernel (self, state=state, mo=mo, ci=ci,
            si=si, d2f=d2f, veff1=veff1, veff2=veff2, _freeze_is=_freeze_is,
            **kwargs)

    def pack_uniq_var (self, xorb, xci, xis=None):
        x = sacasscf_grad.Gradients.pack_uniq_var (self, xorb, xci)
        if xis is not None: x = np.append (x, xis)
        return x

    def unpack_uniq_var (self, x):
        ngorb, nci, nroots, nis = self.ngorb, self.nci, self.nroots, self.nis
        x, xis = x[:ngorb+nci], x[ngorb+nci:]
        xorb, xci = sacasscf_grad.Gradients.unpack_uniq_var (self, x)
        if len (xis)==nis: return xorb, xci, xis
        return xorb, xci

    def _get_is_component (self, xci, ci=None, symm=-1):
        # TODO: state-average-mix
        if ci is None: ci = self.base.ci
        nroots = self.nroots
        xci = np.asarray (xci).reshape (nroots, -1)
        ci = np.asarray (ci).reshape (nroots, -1)
        xis = np.dot (xci.conj (), ci.T)
        if symm > -1: xis -= xis.T
        else:
            assert (np.amax (np.abs (xis + xis.T)) < 1e-8), '{}'.format (xis)
        return xis[np.tril_indices (nroots, k=-1)]

    def _separate_is_component (self, xci, ci=None, symm=-1):
        # TODO: state-average-mix
        is_list = isinstance (xci, list)
        is_tuple = isinstance (xci, tuple)
        if ci is None: ci = self.base.ci
        nroots = self.nroots
        ishape = np.asarray (xci).shape
        xci = np.asarray (xci).reshape (nroots, -1)
        xci = np.asarray (xci).reshape (nroots, -1)
        ci = np.asarray (ci).reshape (nroots, -1)
        xis = np.dot (xci.conj (), ci.T)
        xci -= np.dot (xis.conj (), ci)
        xci = xci.reshape (ishape)
        if is_list: xci = list (xci)
        elif is_tuple: xci = tuple (xci)
        if symm > -1: xis -= xis.T
        #else:
        #    assert (np.amax (np.abs (xis + xis.T)) < 1e-8), '{}'.format (xis)
        xis = xis[np.tril_indices (nroots, k=-1)]
        return xci, xis


    def get_wfn_response (self, si_bra=None, si_ket=None, state=None, mo=None,
            ci=None, si=None, eris=None, veff1=None, veff2=None,
            _freeze_is=False, **kwargs):
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if si is None: si = self.base.si
        if state is None: state = self.state
        if si_bra is None: si_bra = si[:,state]
        if si_ket is None: si_ket = si[:,state]
        log = lib.logger.new_logger (self, self.verbose)
        si_diag = si_bra * si_ket
        nroots, ngorb, nci = self.nroots, self.ngorb, self.nci
        ptr_is = ngorb + nci

        # Diagonal: PDFT component
        nlag = self.nlag-self.nis
        g_all_pdft = np.zeros (nlag)
        for i, (amp, c, v1, v2) in enumerate (zip (si_diag, ci, veff1, veff2)):
            if not amp: continue
            g_i = mcpdft_grad.Gradients.get_wfn_response (self,
                state=i, mo=mo, ci=ci, veff1=v1, veff2=v2, nlag=nlag, **kwargs)
            g_all_pdft += amp * g_i
            if self.verbose >= lib.logger.DEBUG:
                g_orb, g_ci = self.unpack_uniq_var (g_i)
                g_ci, g_is = self._separate_is_component (g_ci, ci=ci, symm=0)
                log.debug ('g_is pdft state {} component:\n{} * {}'.format (i,
                    amp, g_is))

        # DEBUG
        g_orb_pdft, g_ci = self.unpack_uniq_var (g_all_pdft)
        g_ci, g_is_pdft = self._separate_is_component (g_ci, ci=ci, symm=0)

        # Off-diagonal: heff component
        g_orb_heff, g_is_heff = sipdft_heff_response (self, mo=mo, ci=ci,
            si_bra=si_bra, si_ket=si_ket, eris=eris)

        log.debug ('g_is pdft total component:\n{}'.format (g_is_pdft))
        log.debug ('g_is heff component:\n{}'.format (g_is_heff))

        # Combine
        g_orb = g_orb_pdft + g_orb_heff
        g_is = g_is_pdft + g_is_heff
        if _freeze_is: g_is[:] = 0.0
        g_all = self.pack_uniq_var (g_orb, g_ci, g_is)

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
        ham_od = self.base.ham_si.copy ()
        ham_od[np.diag_indices (self.nroots)] = 0.0
        ham_od += ham_od.T # This corresponds to the arbitrary newton_casscf * 2
        fcasscf = self.make_fcasscf_sa ()
        hop, Adiag = newton_casscf.gen_g_hop (fcasscf, mo, ci, eris, verbose)[2:]
        ngorb, nci = self.ngorb, self.nci
        # TODO: cacheing sarot_response? or an x=0 branch?
        def Aop (x):
            x_v, x_is = x[:ngorb+nci], x[ngorb+nci:]
            Ax_v = hop (x_v) + self.sarot_response (x_is, mo=mo, ci=ci, eris=eris)
            x_c = self.unpack_uniq_var (x_v)[1]
            Ax_is = np.dot (d2f, x_is)
            Ax_o, Ax_c = self.unpack_uniq_var (Ax_v)
            Ax_c, Ax_is2 = self._separate_is_component (Ax_c)
            Ax_c_od = list (np.tensordot (-ham_od, np.stack (x_c, axis=0), axes=1))
            Ax_c = [a1 + (w*a2) for a1, a2, w in zip (Ax_c, Ax_c_od, self.base.weights)]
            #assert (np.amax (np.abs (Ax_is2 - Ax_is)) < 1e-8), '{}\n{}'.format (Ax_is, Ax_is2)
            return self.pack_uniq_var (Ax_o, Ax_c, Ax_is)
        return Aop, Adiag

    def get_lagrange_precond (self, Adiag, level_shift=None, ci=None, d2f=None, **kwargs):
        if level_shift is None: level_shift = self.level_shift
        if ci is None: ci = self.base.ci
        if d2f is None: d2f = self.base.sarot_objfn (ci=ci)[2]
        return SIPDFTLagPrec (Adiag=Adiag, level_shift=level_shift, ci=ci, 
            d2f=d2f, grad_method=self)

    def get_ham_response (self, si_bra=None, si_ket=None, state=None, mo=None,
            ci=None, si=None, eris=None, veff1=None, veff2=None, mf_grad=None, 
            atmlst=None, verbose=None, **kwargs):
        ''' write sipdft heff Hellmann-Feynman calculator; sum over diagonal
            PDFT Hellmann-Feynman terms '''
        if atmlst is None: atmlst = self.atmlst
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if si is None: si = self.base.si
        if state is None: state = self.state
        if si_bra is None: si_bra = si[:,state]
        if si_ket is None: si_ket = si[:,state]
        if mf_grad is None: mf_grad = self.base._scf.nuc_grad_method ()
        if verbose is None: verbose = self.verbose
        si_diag = si_bra * si_ket
        log = logger.new_logger (self, verbose)

        # Fix messed-up counting of the nuclear part
        de_nuc = mf_grad.grad_nuc (self.mol, atmlst)
        log.debug ('SI-PDFT gradient n-n terms:\n{}'.format (de_nuc))
        de = si_diag.sum () * de_nuc.copy ()

        # Diagonal: PDFT component
        for i, (amp, c, v1, v2) in enumerate (zip (si_diag, ci, veff1, veff2)):
            if not amp: continue
            de_i = mcpdft_grad.Gradients.get_ham_response (self, state=i, mo=mo,
                ci=ci, veff1=v1, veff2=v2, eris=eris, mf_grad=mf_grad,
                verbose=0, **kwargs) - de_nuc
            log.debug ('SI-PDFT gradient int-state {} EPDFT terms:\n{}'.format
                (i, de_i))
            log.debug ('Factor for these terms: {}'.format (amp))
            de += amp * de_i
        log.debug ('SI-PDFT gradient diag H-F terms:\n{}'.format (de))

        # Off-diagonal: heff component
        de_o = sipdft_heff_HellmanFeynman (self, mo_coeff=mo, ci=ci,
            si_bra=si_bra, si_ket=si_ket, eris=eris, state=state,
            mf_grad=mf_grad, **kwargs)
        log.debug ('SI-PDFT gradient offdiag H-F terms:\n{}'.format (de_o))
        de += de_o
        
        return de

    def get_LdotJnuc (self, Lvec, atmlst=None, verbose=None, mo=None,
            ci=None, eris=None, mf_grad=None, d2f=None, **kwargs):
        ''' Add the IS component '''
        if atmlst is None: atmlst = self.atmlst
        if verbose is None: verbose = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if eris is None and self.eris is None:
            eris = self.eris = self.base.ao2mo (mo)
        elif eris is None:
            eris = self.eris

        ngorb, nci, nis = self.ngorb, self.nci, self.nis
        Lvec_v, Lvec_is = Lvec[:ngorb+nci], Lvec[ngorb+nci:]

        # Double-check Lvec_v sanity
        Lvec_orb, Lvec_ci = self.unpack_uniq_var (Lvec_v)
        Lvec_is2 = self._get_is_component (Lvec_ci, symm=0)
        assert (np.amax (np.abs (Lvec_is2)) < 1e-8), '{} {}'.format (Lvec_is, Lvec_is2)

        # Orbital and CI components
        de_Lv = sacasscf_grad.Gradients.get_LdotJnuc (self, Lvec_v,
            atmlst=atmlst, verbose=verbose, ci=ci, eris=eris, mf_grad=mf_grad,
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

    def get_lagrange_callback (self, Lvec_last, itvec, geff_op):
        # TODO: state-average-mix
        nroots = self.nroots
        log = logger.new_logger (self, self.verbose)
        if isinstance (self.base.fcisolver, CSFFCISolver):
            transf = self.base.fcisolver.transformer
            def _debug_csfs (xci, tag):
                xci_csf = transf.vec_det2csf (xci, normalize=False)
                xci_p = transf.vec_csf2det (xci_csf, normalize=False)
                xci_bs_norm = linalg.norm (np.concatenate (
                    [x.ravel () - y.ravel () for x, y in zip (xci_p, xci)]))
                log.debug ('Broken-spin |{}| = {}'.format (tag, xci_bs_norm))
        else:
            def _debug_csfs (xci, tag): pass
        def my_call (x):
            itvec[0] += 1
            geff = geff_op (x)
            deltax = x - Lvec_last
            gorb, gci, gis = self.unpack_uniq_var (geff)
            deltaorb, deltaci, deltais = self.unpack_uniq_var (deltax)
            gci_norm = linalg.norm (np.asarray (gci).ravel ())
            deltaci_norm = linalg.norm (np.asarray (deltaci).ravel ())
            logger.info(self, ('Lagrange optimization iteration {}, |gorb| = '
                '{}, |gci| = {}, |gis| = {} |dLorb| = {}, |dLci| = {}, |dLis|'
                ' = {}').format (itvec[0], linalg.norm (gorb),
                gci_norm, linalg.norm (gis), linalg.norm (deltaorb),
                deltaci_norm, linalg.norm (deltais)))
            _debug_csfs (gci, 'gci')
            _debug_csfs (deltaci, 'dLci')
            Lvec_last[:] = x[:]
        return my_call

    def debug_lagrange (self, Lvec, bvec, Aop, Adiag, state=None, mo=None,
        ci=None, d2f=None, verbose=None, eris=None, **kwargs):
        if state is None: state = self.state
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if eris is None: eris = self.base.ao2mo (mo)
        if verbose is None: verbose = self.verbose
        nroots = self.nroots
        log = logger.new_logger (self, verbose)
        if isinstance (self.base.fcisolver, CSFFCISolver):
            transf = self.base.fcisolver.transformer
            def _debug_csfs (xci, label, normalize=False):
                strs, vecs = transf.printable_largest_csf (np.asarray (xci), 10,
                    isdet=True, normalize=normalize, order='C')
                log.debug ('Leading CSFs for %s', label)
                for iroot in range (nroots):
                    log.debug (' Root %d', iroot)
                    for s, v in zip (strs[iroot], vecs[iroot]):
                        log.debug ('  |%s>: %s', s, v)
        else: 
            def _debug_csfs (*args, **kwargs): pass
        def _debug_cispace (xci, label):
            log.debug ('%s', label)
            xci_norm = [np.dot (c.ravel (), c.ravel ()) for c in xci]
            try:
                xci_ss = self.base.fcisolver.states_spin_square (xci, self.base.ncas, self.base.nelecas)[0]
            except AttributeError:
                nelec = sum (_unpack_nelec (self.base.nelecas))
                xci_ss = [spin_square (x, self.base.ncas, ((nelec+m)//2,(nelec-m)//2))[0]
                          for x, m in zip (xci, self.spin_states)]
            xci_ss = [x / max (y, 1e-8) for x, y in zip (xci_ss, xci_norm)]
            xci_multip = [np.sqrt (x+.25) - .5 for x in xci_ss]
            xci_norm = np.sqrt (xci_norm)
            for ix, (norm, ss, multip) in enumerate (zip (xci_norm, xci_ss, xci_multip)):
                log.debug ((' State {} norm = {:.7e} ; <S^2> = {:.7f} ; 2S+1'
                            ' = {:.7f}').format (ix, norm, ss, multip))
            ovlp = np.zeros ((nroots, nroots), dtype=xci[0].dtype)
            for i, j in product (range (nroots), repeat=2):
                if self.spin_states[i] != self.spin_states[j]: continue
                ovlp[i,j] = np.dot (xci[i].ravel (), ci[j].ravel ())
            log.debug (' Overlap matrix with CI array:')
            fmt_str = '  ' + ' '.join (['{:8.1e}',]*nroots)
            for row in ovlp: log.debug (fmt_str.format (*row))
        _debug_csfs (ci, 'CI vector', normalize=True)
        borb, bci, bis = self.unpack_uniq_var (bvec)
        log.debug ('Orbital rotation gradient (b) norm = {:.6e}'.format (linalg.norm (borb)))
        _debug_cispace (bci, 'CI gradient (b)')
        _debug_csfs (bci, 'CI gradient (b)')
        Aorb, Aci = self.unpack_uniq_var (Adiag)
        log.debug ('Orbital rotation Hessian (A) diagonal norm = {:.7e}'.format (linalg.norm (Aorb)))
        _debug_cispace (Aci, 'CI Hessian (A) diagonal')
        _debug_csfs (Aci, 'CI Hessian (A) diagonal')
        Lorb, Lci, Lis = self.unpack_uniq_var (Lvec)
        log.debug ('Orbital rotation Lagrange vector (x) norm = {:.7e}'.format (linalg.norm (Lorb)))
        _debug_cispace (Lci, 'CI Lagrange (x) vector')
        _debug_csfs (Lci, 'CI Lagrange (x) vector')
        log.debug ('SI-PDFT Constraint Jacobian (type = {}) (A):'.format (self.base.sarot_name))
        fmt = ' ' + ' '.join (['{:12.5e}' for i in range (self.nis)])
        for row in d2f: log.debug (fmt.format (*row))
        log.debug (' {:>12s} {:>12s}'.format ('Gradient (b)', 'Vector (x)'))
        for g, v in zip (bis, Lis):
            log.debug (' {:12.5e} {:12.5e}'.format (g, v))
        #log.debug ('FOR JIE:')
        #_debug_csfs (self.unpack_uniq_var (self.sarot_response (np.array ([1.0]), mo=mo, ci=ci, eris=eris))[1], 'A_(IS,CI)')

class SIPDFTLagPrec (sacasscf_grad.SACASLagPrec):
    ''' Solve IS part exactly, then do everything else the same '''

    def __init__(self, Adiag=None, level_shift=None, ci=None, grad_method=None,
            d2f=None, **kwargs):
        sacasscf_grad.SACASLagPrec.__init__(self, Adiag=Adiag,
            level_shift=level_shift, ci=ci, grad_method=grad_method)
        self.grad_method = grad_method
        self.log = logger.new_logger (self.grad_method, self.grad_method.verbose)
        self._init_d2f (d2f=d2f, **kwargs)
        self.verbose = self.grad_method.verbose

    def _init_d2f (self, d2f=None, **kwargs):
        self.d2f=d2f
        self.d2f_evals, self.d2f_evecs = linalg.eigh (d2f)
        self.log.debug ('IS component preconditioner eigenvalues: {}'.format (
            1.0/self.d2f_evals))

    def unpack_uniq_var (self, x):
        return self.grad_method.unpack_uniq_var (x)
    def pack_uniq_var (self, x0, x1, x2=None):
        return self.grad_method.pack_uniq_var (x0, x1, x2)

    def __call__(self, x):
        xorb, xci, xis = self.unpack_uniq_var (x)
        Mxorb = self.orb_prec (xorb)
        Mxci = self.ci_prec (xci)
        Mxis = self.is_prec (xis)
        Mx = self.pack_uniq_var (Mxorb, Mxci, Mxis)
        return self.pack_uniq_var (Mxorb, Mxci, Mxis)

    def is_prec (self, xis): 
        Mxis = linalg.solve (self.d2f, xis)
        #if self.verbose >= logger.DEBUG:
        #    x = np.dot (xis, self.d2f_evecs)
        #    Mxis_test = x/self.d2f_evals
        #    Mxis_test = np.dot (self.d2f_evecs, Mxis_test)
        #    Mxis_err = vector_error (Mxis_test, Mxis)[0]
        #    print (Mxis)
        #    print (Mxis_test)
        #    print (Mxis_err)
        #    assert (np.abs (Mxis_err) == 0.0), '{} {}'.format (x, self.d2f_evals)
        return Mxis

if __name__ == '__main__':
    # Test sipdft_heff_response and sipdft_heff_HellmannFeynman by trying to
    # reproduce SA-CASSCF derivatives in an arbitrary basis
    import math
    from pyscf import scf, gto, mcscf
    from mrh.my_pyscf.fci import csf_solver
    xyz = '''O  0.00000000   0.08111156   0.00000000
             H  0.78620605   0.66349738   0.00000000
             H -0.78620605   0.66349738   0.00000000'''
    mol = gto.M (atom=xyz, basis='6-31g', symmetry=False, output='sipdft.log',
        verbose=lib.logger.DEBUG)
    mf = scf.RHF (mol).run ()
    mc = mcpdft.CASSCF (mf, 'tPBE', 4, 4).set (fcisolver = csf_solver (mol, 1))
    mc = mc.state_interaction ([1.0/3,]*3, 'cms').run ()
    mc_grad = Gradients (mc) 
    de = np.stack ([mc_grad.kernel (state=i) for i in range (3)], axis=0)
    print (de)

    #e_states = mc.e_states.copy ()
    #ham_si = np.diag (mc.e_states)
    #
    #mc_grad = mc.nuc_grad_method ()
    #mf_grad = mf.nuc_grad_method ()
    #dh_ref = np.stack ([mc_grad.get_ham_response (state=i) for i in range (3)], axis=0)
    #dw_ref = np.stack ([mc_grad.get_wfn_response (state=i) for i in range (3)], axis=0)
    #dworb_ref, dwci_ref = dw_ref[:,:mc_grad.ngorb], dw_ref[:,mc_grad.ngorb:]
    #assert (linalg.norm (dwci_ref) < 1e-8)

    #si = np.zeros ((3,3), dtype=mc.ci[0].dtype)
    #np.random.seed (0)
    #si[np.tril_indices (3, k=-1)] = math.pi * (np.random.rand ((3)) - 0.5)
    #si = linalg.expm (si-si.T)
    #ci_arr = np.asarray (mc.ci)
    #ham_si = si @ ham_si @ si.T
    #e_mcscf = ham_si.diagonal ()
    #eris = mc.ao2mo (mc.mo_coeff)
    #ci = mc.ci = mc_grad.ci = mc_grad.base.ci = list (np.tensordot (si, ci_arr, axes=1))
    #ci_arr = np.asarray (mc.ci)

    #dh_diag = np.stack ([mc_grad.get_ham_response (state=i, ci=ci) for i in range (3)], axis=0)
    #dw_diag = np.stack ([mc_grad.get_wfn_response (state=i, ci=ci) for i in range (3)], axis=0)
    #si_diag = si * si
    #dh_diag = np.einsum ('sac,sr->rac', dh_diag, si_diag)
    #dworb_diag, dwci_diag = dw_diag[:,:mc_grad.ngorb], dw_diag[:,mc_grad.ngorb:]
    #dworb_diag = np.einsum ('sc,sr->rc', dworb_diag, si_diag)
    #dwci_diag = np.einsum ('rpab,qab->rpq', dwci_diag.reshape (3,3,6,6), ci_arr)
    #dwci_diag -= dwci_diag.transpose (0,2,1)
    #dwci_diag = np.einsum ('spq,sr->rpq', dwci_diag, si_diag)

    #dh_offdiag = np.zeros_like (dh_diag)
    #dw_offdiag = np.zeros_like (dw_diag)
    #for i in range (3):
    #    dw_offdiag[i] += sipdft_heff_response (mc_grad, ci=ci, state=i, eris=eris,
    #        si_bra=si[:,i], si_ket=si[:,i], ham_si=ham_si, e_mcscf=e_mcscf)
    #    dh_offdiag[i] += sipdft_heff_HellmanFeynman (mc_grad, ci=ci, state=i,
    #        si_bra=si[:,i], si_ket=si[:,i], eris=eris)
    #dworb_offdiag, dwci_offdiag = dw_offdiag[:,:mc_grad.ngorb], dw_offdiag[:,mc_grad.ngorb:]
    #dwci_offdiag = np.einsum ('rpab,qab->rpq', dwci_offdiag.reshape (3,3,6,6), ci_arr)
 
    #dworb_test = dworb_diag + dworb_offdiag
    #dwci_test = dwci_diag + dwci_offdiag
    #dh_test = dh_diag + dh_offdiag

    #dworb_err = dworb_test - dworb_ref
    #dwci_err = dwci_test 
    #dh_err = dh_test - dh_ref 
    #print ("test", linalg.norm (dworb_err), linalg.norm (dwci_err), linalg.norm (dh_err))

