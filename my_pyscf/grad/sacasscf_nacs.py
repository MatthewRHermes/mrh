import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.fci import direct_spin1
from pyscf.mcscf import newton_casscf

# The extension from gradients -> NACs has three basic steps:
# 0. ("state" index integer -> tuple)
# 1. fcisolver.make_rdm12 -> fcisolver.trans_rdm12
# 2. remove core-orbital and nuclear contributions to everything
# 3. option to include the "csf contribution" 
# Additional good ideas:
# a. Option to multiply NACs by the energy difference to control
#    singularities

def _unpack_state (state):
    if hasattr (state, '__len__'): return state[0], state[1]
    return state, state

def grad_elec_core (mc_grad, mo_coeff=None, atmlst=None, eris=None,
                    mf_grad=None):
    '''Compute the core-electron part of the CASSCF (Hellmann-Feynman)
    gradient using a modified RHF grad_elec call.'''
    mc = mc_grad.base
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if eris is None: eris = mc.ao2mo (mo_coeff)
    if mf_grad is None: mf_grad = mc._scf.nuc_grad_method ()
    ncore = mc.ncore
    moH = mo_coeff.conj ().T
    f0 = (moH @ mc.get_hcore () @ mo_coeff) + eris.vhf_c
    mo_energy = f0.diagonal ().copy ()
    mo_occ = np.zeros_like (mo_energy)
    mo_occ[:ncore] = 2.0
    f0 *= mo_occ[None,:]
    dme0 = lambda * args: mo_coeff @ ((f0+f0.T)*.5) @ moH
    with lib.temporary_env (mf_grad, make_rdm1e=dme0, verbose=0):
     with lib.temporary_env (mf_grad.base, mo_coeff=mo, mo_occ=mo_occ):
        # Second level there should become unnecessary in future, if anyone
        # ever gets around to cleaning up pyscf.df.grad.rhf & pyscf.grad.rhf
        de = mf_grad.grad_elec (mo_coeff=mo_coeff, mo_energy=mo_energy,
            mo_occ=mo_occ, atmlst=atmlst)
    return de

def grad_elec_active (mc_grad, mo_coeff=None, ci=None, atmlst=None,
                      eris=None, mf_grad=None, verbose=None):
    '''Compute the active-electron part of the CASSCF (Hellmann-Feynman)
    gradient by subtracting the core-electron part.'''
    t0 = (logger.process_clock (), logger.perf_counter ())
    mc = mc_grad.base
    log = logger.new_logger (mc_grad, verbose)
    if mf_grad is None: mf_grad=mc._scf.nuc_grad_method ()
    de = mc_grad.grad_elec (mo_coeff=mo_coeff, ci=ci, atmlst=atmlst,
                            verbose=0)
    de -= grad_elec_core (mc_grad, mo_coeff=mo_coeff, atmlst=atmlst,
                          eris=eris, mf_grad=mf_grad)
    log.debug ('CASSCF active-orbital gradient:\n{}'.format (de))
    log.timer ('CASSCF active-orbital gradient', *t0)
    return de

def gen_g_hop_active (mc, mo, ci0, eris, verbose=None):
    '''Compute the active-electron part of the orbital rotation gradient
    by patching out the appropriate block of eris.vhf_c'''
    ncore = mc.ncore
    vnocore = eris.vhf_c.copy ()
    vnocore[:,:ncore] = -moH @ mc.get_hcore () @ mo[:,:ncore]
    with lib.temporary_env (eris, vhf_c=vnocore):
        return newton_casscf.gen_g_hop (mc, mo, ci0, eris, verbose=verbose)


class NonAdiabaticCouplings (sacasscf_grad.Gradients):
    '''SA-CASSCF non-adiabatic couplings between states

    Extra attributes:

    mult_ediff : logical
        If True, returns NACs multiplied by the energy difference.
        Useful near conical intersections to avoid numerical problems.
    incl_csf : logical
        If True, the NACs include the ``CSF contribution.'' This term
        does not observe translational and rotational symmetry.
    '''

    def __init__(self, mc, state=None, mult_ediff=False, incl_csf=False):
        self.mult_ediff = mult_ediff
        self.incl_csf = incl_csf
        sacasscf_grad.Gradients.__init__(self, mc, state=state)

    def make_fcasscf (self, state=None, casscf_attr=None,
                      fcisolver_attr=None):
        if state is None: state = self.state
        if casscf_attr is None: casscf_attr = {}
        if fcisolver_attr is None: fcisolver_attr = {}
        bra, ket = _unpack_state (state)
        ci, ncas, nelecas = self.base.ci, self.base.ncas, self.base.nelecas
        # TODO: use fcisolver.fcisolvers in state-average mix case for this
        castm1, castm2 = direct_spin1.trans_rdm12 (ci[bra], ci[ket], ncas,
                                                   nelecas)
        castm1 = 0.5 * (castm1 + castm1.T)
        castm2 = 0.5 * (castm2 + castm2.transpose (2,3,0,1))
        fcisolver_attr['make_rdm12'] = lambda *args : castm1, castm2
        fcisolver_attr['make_rdm1'] = lambda *args : castm1
        fcisolver_attr['make_rdm2'] = lambda *args : castm2
        return sacasscf_grad.Gradients.make_fcasscf (
            state=ket, casscf_attr=casscf_attr, fcisolver_attr=fcisolver_attr)


    def get_wfn_response (self, atmlst=None, state=None, verbose=None, mo=None, ci=None, **kwargs):
        if state is None: state = self.state
        if atmlst is None: atmlst = self.atmlst
        if verbose is None: verbose = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        log = logger.new_logger (self, verbose)
        bra, ket = _unpack_state (state)
        fcasscf = self.make_fcasscf (state)
        fcasscf.mo_coeff = mo
        fcasscf.ci = ci[ket]
        eris = fcasscf.ao2mo (mo)
        g_all_ket = gen_g_hop_active (fcasscf, mo, ci[ket], eris, verbose)[0]
        g_all = np.zeros (self.nlag)
        g_all[:self.ngorb] = g_all_ket[:self.ngorb]
        # The fun thing about the ci sector is that you swap them (&/2):
        # <I|[H,|A><I|-|I><A|]|J> = <A|H|J> = <J|[H,|A><J|-|J><A|]|J>/2
        # (It should be zero for converged SA-CASSCF anyway, though)
        g_ci_bra = 0.5 * g_all_ket[self.ngorb:]
        g_all_bra = gen_g_hop_active (fcasscf, mo, ci[bra], eris, verbose)[0]
        g_ci_ket = 0.5 * g_all_bra[self.ngorb:]
        # I have to make sure they don't talk to each other because the
        # preconditioner doesn't explore that space at all. Should I
        # instead solve at the init_guess step, like in MC-PDFT?
        # In practice it should all be zeros but how tightly does
        # everything have to be converged?
        ndet_ket = (self.na_states[ket], self.nb_states[ket])
        ndet_bra = (self.na_states[bra], self.nb_states[bra])
        if ndet_ket==ndet_bra:
            ket2bra = np.dot (ci[bra].conj (), g_ci_ket)
            bra2ket = np.dot (ci[ket].conj (), g_ci_bra)
            log.debug ('SA-CASSCF <bra|H|ket>,<ket|H|bra> check: %5.3g , %5.3g',
                       ket2bra, bra2ket)
            g_ci_ket -= ket2bra * ci[bra]
            g_ci_bra -= bra2ket * ci[ket]
        ndet_ket = ndet_ket[0]*ndet_ket[1]
        ndet_bra = ndet_bra[0]*ndet_bra[1]
        # No need to reshape or anything, just use the magic of repeated slicing
        offs_ket = (sum ([na * nb for na, nb in zip(
                         self.na_states[:ket], self.nb_states[:ket])])
                    if ket > 0 else 0)
        offs_bra = (sum ([na * nb for na, nb in zip(
                         self.na_states[:bra], self.nb_states[:bra])])
                    if ket > 0 else 0)
        g_all[self.ngorb:][offs:][:ndet_ket] = g_ci_ket[self.ngorb:]
        g_all[self.ngorb:][offs:][:ndet_bra] = g_ci_bra[self.ngorb:]
        return g_all


    def get_ham_response (self, state=None, atmlst=None, verbose=None, mo=None,
                          ci=None, eris=None, mf_grad=None, **kwargs):
        if state is None: state = self.state
        if atmlst is None: atmlst = self.atmlst
        if verbose is None: verbose = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if eris is None and self.eris is None:
            eris = self.eris = self.base.ao2mo (mo)
        elif eris is None:
            eris = self.eris
        incl_csf = kwargs.get ('incl_csf', self.incl_csf)
        bra, ket = _unpack_state (state)
        fcasscf_grad = casscf_grad.Gradients (self.make_fcasscf (state))
        nac = grad_elec_active (fcasscf_grad, mo_coeff=mo, ci=ci[ket],
                                atmlst=atmlst, verbose=verbose)
        if incl_csf: nac += self.nac_csf (mo_coeff=mo, ci=ci, state=state,
                                          atmlst=atmlst)
        return nac

    def nac_csf (self, mo_coeff=None, ci=None, state=None, atmlst=None):
        if state is None: state = self.state
        if atmlst is None: atmlst = self.atmlst
        if mo_coeff is None: mo_coeff = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        nac = nac_csf (mo_coeff, ci, state, atmlst)
        bra, ket = _unpack_state (state)
        e_bra = self.base.e_states[bra]
        e_ket = self.base.e_states[ket]
        nac *= e_bra - e_ket
        return nac

    def kernel (self, *args, **kwargs):
        mult_ediff = kwargs.get ('mult_ediff', self.mult_ediff)
        state = kwargs.get ('state', self.state)
        conv, nac = sacasscf_grad.Gradients.kernel (self, *args, **kwargs)
        if not mult_ediff:
            bra, ket = _unpack_state (state)
            e_bra = self.base.e_states[bra]
            e_ket = self.base.e_states[ket]
            nac /= e_bra - e_ket
        return conv, nac


