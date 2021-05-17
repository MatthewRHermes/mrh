import numpy as np
from scipy import linalg
from pyscf import lib
from pyscf.mcscf.addons import StateAverage*
# API for general state-interaction MC-PDFT method object
# In principle, various forms can be implemented: CMS, XMS, etc.

def make_ham_si (mc,ci):
    ''' Build Hamiltonian matrix in basis of ci vector, with diagonal elements
        computed by PDFT and off-diagonal elements computed by MC-SCF '''

    nroots = mc.fcisolver.nroots
    ci = np.asarray(ci)

    e_pdft = np.stack ([mcpdft.mcpdft.kernel (mc, ot=mc.otfnal, ci=c)
        for c in ci], axis=1)
    e_int, e_ot = e_pdft

    h1, h0 = mc.get_h1eff ()
    h2 = mc.get_h2eff ()
    h2eff = direct_spin1.absorb_h1e (h1, h2, mc.ncas, mc.nelecas, 0.5)
    hc_all = [direct_spin1.contract_2e (h2eff, c, mc.ncas, mc.nelecas) for c in ci]
    ham_si = np.tensordot (ci, hc_all, axes=((1,2),(1,2))) 
    e_mcscf = ham_si.diagonal ()
    ham_si[np.diag_indices (nroots)] = e_int[:]
    e_cas = e_mcscf - h0
    ci_flat = ci.reshape (nroots, -1)
    ovlp_si = np.dot (ci_flat.conj (), ci_flat.T)
    return ham_si, ovlp_si, e_mcscf, e_cas, e_ot

def si_newton (mc, ci=None, max_cyc=None, conv_tol=None):

    if ci is None: ci = mc.ci
    if max_cyc is None: max_cyc = getattr (mc, 'max_cyc_sarot', 50)
    if conv_tol is None: conv_tol = getattr (mc, 'conv_tol_sarot', 50)
    log = lib.logger.new_logger (mc, mc.verbose)
    nroots = mc.fcisolver.nroots 
    rows,col = np.tril_indices(nroots,k=-1)
    pairs = len(rows)
    u = np.identity(nroots)
    t = np.zeros((nroots,nroots))
    t_old = np.zeros((nroots,nroots))
    conv = False
    ci_array = np.array(mc.ci)
    ci_old = ci_array
    e_c_old = mc.sarot_obj (ci_array)

    for it in range(max_cyc):
        log.info ("****iter {} ***********".format (it))

#       Form U
        U = linalg.expm(t)

#       Rotate T
        ci_rot = np.tensordot(U,ci_old,1)

        e_c, e_g, e_h = mc.sarot_obj (ci_rot)
        e_c = np.array(e_c)

        log.info ("e_coul = {} ; e_g = {}; e_h = {}".format (e_c, e_g, e_h))
        log.info ("e_coul sum = {} ".format (e_c.sum()))

        log.info ("t = {} ".format (t))

        evals, evecs = linalg.eigh (e_h)
        evecs = np.array(evecs)
        log.info ("Hessian eigenvalues: {}".format (evals))

        for i in range(pairs):
            if 1E-09 > evals[i] and evals[i] > -1E-09:
               log.info ("Hess is singular!")
            if evals[i] > 0 :
                neg = False
                break
            neg = True
        log.info ("Hess diag is neg? {}".format (neg))

        if neg == False :
            for i in range(pairs):
                if evals[i] > 0 :
                    evals[i]=-evals[i]

        diag = np.identity(pairs)*evals
        e_h = np.dot(np.dot(evecs,diag),evecs.T)

        t_add = linalg.solve(e_h,-e_g)
        t[:] = 0
        t[np.tril_indices(t.shape[0], k = -1)] = t_add

        t = t - t.T

        t_old = t.copy()

#       Reset Old Values

        ci_old=ci_rot

        grad_norm = np.linalg.norm(e_g)
        log.info ("grad norm = %f", grad_norm)

        if grad_norm < conv_tol and neg == True:
                conv = True
                break
        e_c_old = e_c
    if conv:
        log.note ("{}-PDFT intermediate state determination CONVERGED".format (mc.sarot_name))
    else:
        log.note (("{}-PDFT intermediate state determination did not converge"
                   " after {} cycles").format (mc.sarot_name, it))

    return ci_rot

class StateInteractionMCPDFTSolver ():
    pass
    # tag

class _SIPDFT (StateInteractionMCPDFTSolver):
    # Metaclass parent

    def __init__(self, mc, sarot_obj, sarot, sarot_name):
        self.__dict__.update (mc.__dict__)
        keys = set (('sarot_obj', 'sarot', 'h_ms', 'si', 'mname', 'max_cycle_sarot', 'conv_tol_sarot'))
        self._sarot_obj = sarot_obj
        self._sarot = sarot
        self.max_cycle_sarot = 50
        self.conv_tol_sarot = 1e-8
        self.si = self.h_ms = None
        self.sarot_name = sarot_name
        self._keys = set ((self.__dict__.keys ())).union (keys)

    @property
    def e_states (self):
        return self._e_states
    @e_states.setter
    def e_states (self, x):
        self._e_states = x
    ''' Unfixed to FCIsolver since SI-PDFT state energies are no longer
        CI solutions '''

    def kernel (self, mo=None, ci=None, **kwargs):
        # I should maybe rethink keeping all this intermediate information
        self._init_ot_grids (self.otfnal.otxc, grids_level=self.grids.level)
        _, _, ci, self.mo_coeff, self.mo_energy = self.mcscf_kernel (mo, ci,
            ci, **kwargs)
        self.ci = self.sarot (ci=ci, **kwargs)
        self.ham_si, self.ovlp_si, self.e_mcscf, self.e_ot, self.e_cas = self.make_ham_si (self.ci)
        self.e_states, self.si = linalg.eigh (self.ham_si)
        # TODO: state_average_mix support
        self.e_tot = np.dot (self.e_states, self.weights)
        return self.e_tot, self.e_ot, self.e_mcscf, self.e_cas, self.ci, self.mo_coeff, self.mo_energy

    def sarot (self, ci=None):
        ''' Obtain intermediate states in the average space '''
        if ci is None: ci = self.ci
        return self._sarot (self, ci)

    def sarot_obj (self, ci=None):
        ''' The value, first, and second-derivative matrix of the objective
            function rendered stationary by the intermediate states. Used
            in gradient calculations and possibly in sarot. '''
        if ci is None: ci = self.ci
        return self._sarot_obj (self, ci)

    def make_ham_si (self, ci=None):
        if ci is None: ci = self.ci
        return make_ham_si (ci)


def get_sarotfns (obj):
    if obj.upper () == 'CMS':
        from mrh.my_pyscf.mcpdft.cmspdft4 import e_coul as sarot_obj
        sarot = si_newton
    else:
        raise RuntimeError ('SI-PDFT type not supported')
    return sarot_obj, sarot

def state_interaction (mc, weights=(0.5,0.5), obj='CMS', **kwargs):
    ''' Build state-interaction MC-PDFT method object

    Args:
        mc : instance of class _PDFT
    
    Kwargs:
        weights : sequence of floats
        obj : objective-function type
            Currently supports only 'cms'

    Returns:
        si : instance of class _SIPDFT '''

    if isinstance (mc, StateInteractionMCPDFTSolver):
        raise RuntimeError ('state_interaction recursion! fix API!')
    if isinstance (mc.fcisolver, StateAverageMixFCISolver):
        raise RuntimeError ('TODO: state-average mix support')
    mcbase_class = mc.__class__
    sarot_obj, sarot = get_sarotfns (obj)


