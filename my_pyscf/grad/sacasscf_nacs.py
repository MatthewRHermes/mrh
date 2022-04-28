



def nac_hellmann_feynman(mc_grad, mo_coeff=None, ci=None, state=None,
                         atmlst=None, eris=None, mf_grad=None, verbose=None):
    mc = mc_grad.base
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    if state is None: state = mc_grad.state
    if eris is None: eris = mc.ao2mo (mo_coeff)
    if mf_grad is None: mf_grad = mc._scf.nuc_grad_method ()
    if verbose is None: verbose = mc_grad.verbose
    log = logger.new_logger (mc_grad, verbose)
    t0 = (logger.process_clock (), logger.perf_counter ())

    fcasscf = mc_grad.make_fcasscf (state)
    fcasscf_grad = fcasscf.nuc_grad_method ()
    de = fcasscf_grad.kernel (mo_coeff=mo_coeff, ci=ci[0], atmlst=atmlst, verbose=0)

    # subtract nuc-nuc and core-core (patching out simplified gfock terms)
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
        dde = mf_grad.kernel (mo_coeff=mo, mo_energy=mo_energy, mo_occ=mo_occ,
            atmlst=atmlst)
    de -= dde

    if getattr (self, 'incl_csf_nac'):
        de += mc_grad.csf_nac (mo_coeff=mo_coeff, ci=ci, state=state, atmlst=atmlst, verbose=verbose)

    log.debug ('SA-CASSCF NACs Hellmann-Feynman contribution:\n{}'.format (de)
    log.timer ('SA-CASSCF NACs Hellmann-Feynman contribution', *t0)
    return de

def csf_nac (mc_grad, mo_coeff=None, ci=None, state=None, atmlst=None,
             verbose=None):
    mc = mc_grad.base
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    if state is None: state = mc_grad.state


