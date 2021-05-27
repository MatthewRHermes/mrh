import numpy as np
from scipy import linalg
from pyscf import lib
from pyscf.mcscf.addons import StateAverageMCSCFSolver, StateAverageMixFCISolver
from pyscf.fci import direct_spin1
from mrh.my_pyscf import mcpdft
# API for general state-interaction MC-PDFT method object
# In principle, various forms can be implemented: CMS, XMS, etc.

def make_ham_si (mc,ci):
    ''' Build Hamiltonian matrix in basis of ci vector, with diagonal elements
        computed by PDFT and off-diagonal elements computed by MC-SCF '''

    ci = np.asarray(ci)
    nroots = ci.shape[0]

    e_pdft = np.stack ([mcpdft.mcpdft.kernel (mc, ot=mc.otfnal, ci=c)
        for c in ci], axis=1)
    e_int, e_ot = e_pdft

    h1, h0 = mc.get_h1eff ()
    h2 = mc.get_h2eff ()
    h2eff = direct_spin1.absorb_h1e (h1, h2, mc.ncas, mc.nelecas, 0.5)
    hc_all = [direct_spin1.contract_2e (h2eff, c, mc.ncas, mc.nelecas) for c in ci]
    ham_si = np.tensordot (ci, hc_all, axes=((1,2),(1,2))) 
    e_cas = ham_si.diagonal ().copy ()
    e_mcscf = e_cas + h0
    ham_si[np.diag_indices (nroots)] = e_int[:].copy ()
    ci_flat = ci.reshape (nroots, -1)
    ovlp_si = np.dot (ci_flat.conj (), ci_flat.T)
    return ham_si, ovlp_si, e_mcscf, e_cas, e_ot

def si_newton (mc, ci=None, max_cyc=None, conv_tol=None):

    if ci is None: ci = mc.ci
    if max_cyc is None: max_cyc = getattr (mc, 'max_cyc_sarot', 50)
    if conv_tol is None: conv_tol = getattr (mc, 'conv_tol_sarot', 1e-8)
    log = lib.logger.new_logger (mc, mc.verbose)
    nroots = mc.fcisolver.nroots 
    rows,col = np.tril_indices(nroots,k=-1)
    npairs = nroots * (nroots - 1) // 2
    t = np.zeros((nroots,nroots))
    conv = False

    for it in range(max_cyc):
        log.info ("****iter {} ***********".format (it))

#       Form U
        U = linalg.expm(t)

#       Rotate T
        try:
            ci = np.tensordot(U, ci, 1)
        except ValueError as e:
            print (U.shape, ci.shape)
            raise (e)

        f, df, d2f = mc.sarot_objfn (ci)
        log.info ("e_coul sum = {} ".format (f))
        log.info ("t = {} ".format (t))

        evals, evecs = linalg.eigh (d2f)
        evecs = np.array(evecs)
        log.info ("Hessian eigenvalues: {}".format (evals))

        for i in range(npairs):
            if 1E-09 > evals[i] and evals[i] > -1E-09:
               log.info ("Hess is singular!")
            if evals[i] > 0 :
                neg = False
                break
            neg = True
        log.info ("Hess diag is neg? {}".format (neg))

        if neg == False :
            for i in range(npairs):
                if evals[i] > 0 :
                    evals[i]=-evals[i]

        diag = np.identity(npairs)*evals
        d2f = np.dot(np.dot(evecs,diag),evecs.T)

        Dt = linalg.solve(d2f,-df)
        t[:] = 0
        t[np.tril_indices(t.shape[0], k = -1)] = Dt

        t = t - t.T

        grad_norm = np.linalg.norm(df)
        log.info ("grad = {}".format (df))
        log.info ("grad norm = %f", grad_norm)

        if grad_norm < conv_tol and neg == True:
                conv = True
                break
    if conv:
        log.note ("{}-PDFT intermediate state determination CONVERGED".format (mc.sarot_name))
    else:
        log.note (("{}-PDFT intermediate state determination did not converge"
                   " after {} cycles").format (mc.sarot_name, it))

    return ci

class StateInteractionMCPDFTSolver ():
    pass
    # tag

class _SIPDFT (StateInteractionMCPDFTSolver):
    ''' I'm not going to use subclass to distinguish between various SI-PDFT
        types. Instead, I'm going to use three method attributes:

        _sarot_objfn : callable
            Args: ci vectors
            Returns: float, array (nroots), array (nroots,nroots)
            The value, first, and second derivatives of the objective function
            defining the method. May be used both in performing the SI-PDFT
            energy calculation and in gradients.

        _sarot : callable
            Args: ci vectors
            Returns: ci vectors
            Obtain the intermediate states from the reference states

        sarot_name: string
            Label for I/O.
    '''

    # Metaclass parent

    def __init__(self, mc, sarot_objfn, sarot, sarot_name):
        self.__dict__.update (mc.__dict__)
        keys = set (('sarot_objfn', 'sarot', 'sarot_name', 'ham_si', 'si', 'max_cycle_sarot', 'conv_tol_sarot'))
        self._sarot_objfn = sarot_objfn
        self._sarot = sarot
        self.max_cycle_sarot = 50
        self.conv_tol_sarot = 1e-8
        self.si = self.ham_si = None
        self.sarot_name = sarot_name
        self._keys = set ((self.__dict__.keys ())).union (keys)

    @property
    def e_states (self):
        return getattr (self, '_e_states', self.fcisolver.e_states)
    @e_states.setter
    def e_states (self, x):
        self._e_states = x
    ''' Unfixed to FCIsolver since SI-PDFT state energies are no longer
        CI solutions '''

    def kernel (self, mo=None, ci=None, **kwargs):
        # I should maybe rethink keeping all this intermediate information
        self._init_ot_grids (self.otfnal.otxc, grids_level=self.grids.level)
        ci, self.mo_coeff, self.mo_energy = super().kernel (mo, ci, **kwargs)[-3:]
        self.ci = self.sarot (ci=ci, **kwargs)
        self.ham_si, self.ovlp_si, self.e_mcscf, self.e_ot, self.e_cas = self.make_ham_si (self.ci)
        self._log_sarot ()
        self.e_states, self.si = self._eig_si (self.ham_si)
        # TODO: state_average_mix support
        self.e_tot = np.dot (self.e_states, self.weights)
        self._log_si ()
        return self.e_tot, self.e_ot, self.e_mcscf, self.e_cas, self.ci, self.mo_coeff, self.mo_energy

    # All of the below probably need to be wrapped over solvers in state-interaction-mix metaclass

    def sarot (self, ci=None):
        ''' Obtain intermediate states in the average space '''
        if ci is None: ci = self.ci
        return self._sarot (self, ci)

    def sarot_objfn (self, ci=None):
        ''' The value, first, and second-derivative matrix of the objective
            function rendered stationary by the intermediate states. Used
            in gradient calculations and possibly in sarot. '''
        if ci is None: ci = self.ci
        return self._sarot_objfn (self, ci)

    def _eig_si (self, ham_si):
        return linalg.eigh (ham_si)

    def make_ham_si (self, ci=None):
        if ci is None: ci = self.ci
        return make_ham_si (self, ci)

    def _log_sarot (self):
        e_pdft = self.ham_si.diagonal ()
        nroots = len (e_pdft)
        log = lib.logger.new_logger (self, self.verbose)
        f, df, d2f = self.sarot_objfn ()
        log.note ('%s-PDFT intermediate objective function  value = %.15g  |grad| = %.7g',
            self.sarot_name, f, linalg.norm (df))
        log.note ('%s-PDFT intermediate average energy  EPDFT = %.15g  EMCSCF = %.15g',
            self.sarot_name, np.dot (self.weights, e_pdft),
            np.dot (self.weights, self.e_mcscf))
        log.note ('%s-PDFT intermediate states:', self.sarot_name)
        if getattr (self.fcisolver, 'spin_square', None):
            ss = self.fcisolver.states_spin_square (self.ci, self.ncas,
                                                    self.nelecas)[0]
            for i in range (nroots):
                log.note ('  State %d weight %g  EPDFT = %.15g  EMCSCF = %.15g  S^2 = %.7f',
                    i, self.weights[i], e_pdft[i], self.e_mcscf[i], ss[i])
        else:
            for i in range (nroots):
                log.note ('  State %d weight %g  EPDFT = %.15g  EMCSCF = %.15g',
                    i, self.weights[i], e_pdft[i], self.e_mcscf[i])
        log.info ('Intermediate state Hamiltonian matrix:')
        fmt_str = ' '.join (['{:9.5f}',]*nroots)
        for row in self.ham_si: log.info (fmt_str.format (*row))
        log.info ('Intermediate state overlap matrix:')
        for row in self.ovlp_si: log.info (fmt_str.format (*row))

    def _log_si (self):
        ''' Information about the final states '''
        log = lib.logger.new_logger (self, self.verbose)
        e_pdft = self.e_states
        nroots = len (e_pdft)
        ham_ci = self.ham_si.copy ()
        ham_ci[np.diag_indices (nroots)] = self.e_mcscf.copy ()
        e_mcscf = (np.dot (ham_ci, self.si) * self.si.conj ()).sum (0)
        log.note ('%s-PDFT final states:', self.sarot_name) 
        if getattr (self.fcisolver, 'spin_square', None):
            ci = np.tensordot (self.si.T, np.asarray (self.ci), axes=1)
            ss = self.fcisolver.states_spin_square (ci, self.ncas,
                                                    self.nelecas)[0]
            for i in range (nroots):
                log.note ('  State %d weight %g  EPDFT = %.15g  EMCSCF = %.15g  S^2 = %.7f',
                    i, self.weights[i], e_pdft[i], self.e_mcscf[i], ss[i])
        else:
            for i in range (nroots):
                log.note ('  State %d weight %g  EPDFT = %.15g  EMCSCF = %.15g',
                    i, self.weights[i], e_pdft[i], self.e_mcscf[i])

def get_sarotfns (obj):
    if obj.upper () == 'CMS':
        from mrh.my_pyscf.mcpdft.cmspdft4 import e_coul as sarot_objfn
        sarot = si_newton
    else:
        raise RuntimeError ('SI-PDFT type not supported')
    return sarot_objfn, sarot

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
        raise RuntimeError ('state_interaction recursion! possible API bug!')
    if isinstance (mc.fcisolver, StateAverageMixFCISolver):
        raise RuntimeError ('TODO: state-average mix support')
    if not isinstance (mc, StateAverageMCSCFSolver):
        mc = mc.state_average (weights=weights, **kwargs)
    mcbase_class = mc.__class__
    sarot_objfn, sarot = get_sarotfns (obj)

    class SIPDFT (_SIPDFT, mcbase_class):
        pass
    return SIPDFT (mc, sarot_objfn, sarot, obj)
    

if __name__ == '__main__':
    # This ^ is a convenient way to debug code that you are working on. The
    # code in this block will only execute if you run this python script as the
    # input directly: "python cmspdft3.py".

    from pyscf import scf, gto
    from mrh.my_pyscf.tools import molden # My version is better for MC-SCF
    from mrh.my_pyscf.fci import csf_solver
    xyz = '''O  0.00000000   0.08111156   0.00000000
             H  0.78620605   0.66349738   0.00000000
             H -0.78620605   0.66349738   0.00000000'''
    mol = gto.M (atom=xyz, basis='sto-3g', symmetry=False, output='sipdft.log', verbose=lib.logger.DEBUG)
    mf = scf.RHF (mol).run ()
    mc = mcpdft.CASSCF (mf, 'tPBE', 4, 4).set (fcisolver = csf_solver (mol, 1))
    mc = mc.state_interaction ([1.0/3,]*3, 'cms').run ()


