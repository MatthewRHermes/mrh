import numpy as np
import time
from itertools import product
from scipy import linalg
from pyscf import gto, dft, ao2mo, fci, mcscf, lib
from pyscf.lib import logger, temporary_env
from pyscf.mcscf import mc_ao2mo
from pyscf.mcscf.addons import StateAverageMCSCFSolver, state_average_mix, state_average_mix_, state_average
from mrh.util.rdm import get_2CDM_from_2RDM, get_2CDMs_from_2RDMs
from mrh.my_pyscf.mcpdft import pdft_veff
from mrh.my_pyscf.mcpdft.otpd import get_ontop_pair_density
from mrh.my_pyscf.mcpdft.otfnal import otfnal, transfnal, ftransfnal
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf.grad.mcpdft import Gradients
# MRH: this ^ import can cause recursive inheritance crashes if it appears
# higher in this list, before other mcpdft modules. I need to fix it but in the
# meantime just make sure it's the last import and it should be OK.

def get_cmspdft_intermediate_states (mc, mo_coeff=None, ci=None):
    ''' Compute the CI vectors which maximize the CMS-PDFT intermediate
        objective function: Qa-a = sum_state (e_coul[state])

        Args:
            mc : instance of mcpdft.mcpdft.StateAverageMCPDFTSolver class

        Kwargs:
            mo_coeff : ndarray of shape (nao, nmo)
                Contains molecular orbital coefficients
            ci : sequence of overall shape (nroots, ndeta, ndetb)
                Contains CI vectors of reference states

        Returns:
            conv : logical
                Reports whether the optimization successfully converged
            e_int : ndarray of shape (nroots)
                Total MC-PDFT energies of intermediate states
            ci_int : like ci
                Contains CI vectors of CMS-PDFT intermediate states
    '''

    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    # ^ These are the ultimate controlling variables. We always want to give
    # the user and future developers the option of computing this with these.
    # OTOH, "get_cmspdft_intermediate_states (mc)" is more convenient API most
    # of the time, so mo_coeff and ci are optional.
    ci_ref = np.asarray (ci)
    nroots = ci_ref.shape[0]
    t0 = (time.clock (), time.time ())

#Initialization
    log = lib.logger.new_logger (mc, mc.verbose)
    log.info ("Entering get_cmspdft_intermediate_states")
    w = first_quantization_coulomb (mc, mo_coeff=mo_coeff, ci=ci_ref)

#Print the First Coulomb Energy
    e_coul, Q = get_Q_dQ_d2Q (mc, w=w)[:2]
    log.debug ("Reference state e_coul {}".format (e_coul)) 
    log.info ("Reference state e_coul sum = %f", Q)

#Loop Initializations
    maxiter = 50
    thrs = 1.0e-06
    Q_old = Q
    conv = False
    t1 = (time.clock (), time.time ())
    U = np.eye (nroots, dtype=ci_ref.dtype)
    t = np.zeros((nroots,nroots))

#################
#Begin Loop 
#################

# MRH: it's good to break these things down into smaller problems. Put most
# steps in the loop in their own separate functions.

    for it in range(maxiter):
        log.info ("****iter {} ***********".format (it))
       
#       Form U
        U = np.dot (linalg.expm(t), U)

#       Rotate w
        w_rot = lib.einsum ('pk,ql,rm,sn,klmn->pqrs', U, U, U, U, w)

#       Objective function and derivatives
        e_coul, Q, grad, hess = get_Q_dQ_d2Q (mc, w=w_rot)
        log.info ("Sum e_coul = {} ; difference = {}".format (Q, Q-Q_old)) 

#       Convergence check and step
        t = _cmspdft_step (grad, hess, log, thrs)
        conv = (t is None)
        if conv: break

#       Reset Old Values
        Q_old = Q

#########################
# End Loop

    ci_int = np.tensordot (U, ci_ref, axes=1)
    t1 = log.timer ('CMS-PDFT loop', *t1)
    if conv:
        log.note (("CMS-PDFT intermediate state determination CONVERGED"
                   " after {} cycles").format (it+1))
    else:
        log.note (("CMS-PDFT intermediate state determination did not converge"
                   " after {} cycles").format (it+1))

# Intermediate Energies 
# Run MC-PDFT
    e_int = np.asarray ([mcpdft.mcpdft.energy_tot (mc, ot=mc.otfnal, ci=c)[0] 
        for c in ci_int])

# (OPTIONAL) sort states in ascending order by their intermediate energies
    idx_sort = np.argsort (e_int)
    e_int = e_int[idx_sort]
    ci_int = ci_int[idx_sort,...]

    log.info ("CMS-PDFT intermediate state energies: {}".format (e_int))
    log.timer ('CMS-PDFT intermediate state optimization', *t0)

    return conv, e_int, list(ci_int)


def get_Q_dQ_d2Q (mc, mo_coeff=None, ci=None, w=None):
    ''' Compute the Coulomb energies, their sum, and the first and second
    derivatives of their sum for a set of states in a state-average MC-SCF
    context.

    Args:
        mc : instance of mcscf.StateAverageMCSCFSolver class

    Kwargs:
        mo_coeff : ndarray of shape (nao, nmo)
            Molecular-orbital coefficients
        ci : sequence of shape (nroots, ndeta, ndetb)
            CI vectors defining nroots states
        w : ndarray of shape [nroots,]*4
            Contains "first-quantization Coulomb" array
            see cmspdft3.first_quantization_coulomb

    Returns:
        e_coul : list of length (nroots)
            Coulomb energies of states "ci"
        Q : float
            sum of e_coul
        dQ : ndarray of shape (nroots*(nroots-1)//2)
            First derivative of Q along state-rotation coordinates
        d2Q : ndarray of shape (len (dQ), len (dQ))
            Second derivative of Q along state-rotation coordinates
    '''

    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    ci = np.asarray (ci)
    # ^ These are the ultimate controlling variables. We always want to give
    # the user and future developers the option of computing this with these
    # alone. OTOH, during a self-consistent cycle, it makes more sense to
    # precompute w, so this optional variable overrides the others if provided.
    if w is None: w = first_quantization_coulomb (mc, mo_coeff=mo_coeff, ci=ci,
        eri_cas=eri_cas)

    nroots = w.shape[0]
    row, col = np.tril_indices (nroots, k=-1)

    # I don't usually use einsum because it's slow, but nroots does not
    # automatically scale with the size of the molecule so in this context the
    # computational cost is worth the simplicity

    ecoul = np.einsum ('kkkk->k', w) / 2
    Q = ecoul.sum ()
    dQ = np.einsum ('klkk->kl', w) - np.einsum ('klll->kl', w)
    dQ = 2 * dQ[row,col]

    v_delta_lm =  (np.einsum ('knkk->kn', w)[:,None,:]
               -   np.einsum ('knll->kln', w)
               +   np.einsum ('nknn->kn', w)[:,None,:]
               -   np.einsum ('knmm->kmn', w)
               - 4*np.einsum ('kllm->klm', w))
    d2Q = np.zeros_like (w)
    for l in range (nroots): d2Q[:,l,l,:] = v_delta_lm[:,l,:]
    d2Q -= d2Q.transpose (1,0,2,3)
    d2Q -= d2Q.transpose (0,1,3,2)
    d2Q = d2Q[row,col,:,:][:,row,col]
    return ecoul, Q, dQ, d2Q

def first_quantization_coulomb (mc, mo_coeff=None, ci=None, eri_cas=None):
    ''' Generate the "first-quantization coulomb" array: a 4-dimensional array
        int <k|rho|l>(1/r12)<m|rho|n> dr, where klmn index CI vectors in a CAS.
        Used for CMS-PDFT '''
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    ci = np.asarray (ci)
    ncore, ncas, nelecas = mc.ncore, mc.ncas, mc.nelecas
    nocc = ncore + ncas
    if eri_cas is None: 
        mo_cas = mo_coeff[:,ncore:nocc]
        eri_cas = mc.get_h2eff (mo_cas)
    eri_cas = ao2mo.restore (1, eri_cas, ncas)
    # ^ very powerful function; makes sure eri_cas is completely unpacked no 
    # matter what format it starts out in

    # Density-matrix throat clearing
    nroots = len (ci)
    npairs = nroots * (nroots-1) // 2
    row, col = np.tril_indices (nroots, k=0)
    gendm1 = np.asarray (mc.fcisolver.states_trans_rdm12 (ci[col], ci[row], 
        ncas, nelecas)[0])
    # ^ The diagonal cases of these are equal to rdms! No need to use two
    # separate functions!

    w = lib.einsum ('kpq,pqrs,lrs->kl', gendm1, eri_cas, gendm1)
    w = ao2mo.restore (1, w, nroots)
    # ^ This array has the same symmetry as the ERI array, and the tril_indices
    # packed it in exactly the same way, so we can use this to recover all
    # four indices

    return w
    
def _cmspdft_step (grad, hess, log, thrs):
    nroots = len (grad)
    grad_norm = linalg.norm (grad)
    log.debug ("grad: {}".format (grad))
    log.info ("grad norm = %e", grad_norm)
    evals, evecs = linalg.eigh (hess)
    log.info ("Hessian eigenvalues: {}".format (evals))
    if (grad_norm<thrs) and (np.all (evals<0)): return None

#   (OPTIONAL) Kick away from minima to accelerate convergence
    grad = np.dot (evecs.T, grad)
    idx_minima = (evals>0) & (np.abs (grad)<thrs)
    grad_sign = np.logical_or (np.sign (grad[idx_minima]), 1)
    grad_shift = grad_sign * thrs # ??? how much ???
    grad[idx_minima] += grad_shift
    grad = np.dot (evecs, grad)

#   Effective Hessian that moves us away from minima
    evals = -np.abs (evals) 
    hess = np.dot (evecs * evals[None,:], evecs.T)

#   Rotation step
    t_add = linalg.solve(hess,-grad)
    t = np.zeros ((nroots, nroots), dtype=grad.dtype)
    t[np.tril_indices(nroots, k = -1)] = t_add
    t = t - t.T

    return t
    
    

if __name__ == '__main__':
    # This ^ is a convenient way to debug code that you are working on. The
    # code in this block will only execute if you run this python script as the
    # input directly: "python cmspdft3.py".

    from pyscf import scf
    from mrh.my_pyscf.tools import molden # My version is better for MC-SCF
    from mrh.my_pyscf.fci import csf_solver
    xyz = '''O  0.00000000   0.08111156   0.00000000
             H  0.78620605   0.66349738   0.00000000
             H -0.78620605   0.66349738   0.00000000'''
    mol = gto.M (atom=xyz, basis='sto-3g', symmetry=False, output='cmspdft3.log', verbose=lib.logger.DEBUG)
    mf = scf.RHF (mol).run ()
    mc = mcpdft.CASSCF (mf, 'tPBE', 4, 4).set (fcisolver = csf_solver (mol, 1))
    mc = mc.state_average ([1.0/3,]*3).run ()
    molden.from_sa_mcscf (mc, 'h2o_sapdft_sa.molden', cas_natorb=True)
    # ^ molden file with state-averaged NOs
    for i in range (3):
        fname = 'h2o_sapdft_ref{}.molden'.format (i)
        # ^ molden file with ith reference state NOs
        molden.from_sa_mcscf (mc, fname, state=i, cas_natorb=True)

    conv, E_int, ci_int = get_cmspdft_intermediate_states (mc)
    print ("The iteration did{} converge".format ((' not','')[int (conv)]))
    print ("The intermediate-state energies are",E_int)
    print (("Molden files with intermediate-state NOs are in "
            "h2o_sapdft_int?.molden"))
    with lib.temporary_env (mc, ci=ci_int):
        # ^ See line 258
        for i in range (3):
            fname = 'h2o_sapdft_int{}.molden'.format (i)
            molden.from_sa_mcscf (mc, fname, state=i, cas_natorb=True)
    
    print ("See cmspdft3.log for more information")

