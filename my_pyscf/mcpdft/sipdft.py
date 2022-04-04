import numpy as np
from scipy import linalg
from pyscf import lib
from pyscf.mcscf.addons import StateAverageMCSCFSolver
from pyscf.mcscf.addons import StateAverageMixFCISolver
from pyscf.fci import direct_spin1
from mrh.my_pyscf import mcpdft

# API for general state-interaction MC-PDFT method object
# In principle, various forms can be implemented: CMS, XMS, etc.

# API cleanup desiderata:
# 1. "sipdft", "state_interaction" -> "mspdft", "multi_state"
# 2. "sarot", "sarot_name" -> "diabatize", "diabatization"
# 3. "get_ci_mcscf" and "get_ci_final" QOL functions
# 4. Canonicalize function to quickly generate mo_coeff, ci, mo_occ, mo_energy
#    for different choices of intermediate, reference, final states.
# 5. Probably "_finalize" stuff
# 6. checkpoint stuff
def make_heff_mcscf (mc, mo_coeff=None, ci=None):
    ''' Build Hamiltonian matrix in basis of ci vector

        Args:
            mc : an instance of MCPDFT class

        Kwargs:
            mo_coeff : ndarray of shape (nao, nmo)
                MO coefficients
            ci : ndarray or list of len (nroots)
                CI vectors describing the model space, presumed to be in
                the optimized intermediate-state basis

        Returns:
            heff_mcscf : ndarray of shape (nroots, nroots)
                Effective MC-SCF hamiltonian matrix in the basis of the
                provided CI vectors
    '''
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci

    ci = np.asarray(ci)
    nroots = ci.shape[0]

    h1, h0 = mc.get_h1eff (mo_coeff)
    h2 = mc.get_h2eff (mo_coeff)
    h2eff = direct_spin1.absorb_h1e (h1, h2, mc.ncas, mc.nelecas, 0.5)
    hc_all = [direct_spin1.contract_2e (h2eff, c, mc.ncas, mc.nelecas)
        for c in ci]
    heff = np.tensordot (ci, hc_all, axes=((1,2),(1,2)))
    idx = np.diag_indices_from (heff)
    heff[idx] += h0
    return heff

def si_newton (mc, ci=None, objfn=None, max_cyc=None, conv_tol=None,
        sing_tol=None, nudge_tol=None):
    ''' Optimize the intermediate states describing the model space of
        an MS-PDFT calculation by maximizing the provided objective
        function using a gradient-ascent algorithm

        Args:
            mc : an instance of SIPDFT class

        Kwargs:
            ci : ndarray or list of len (nroots)
                CI vectors spanning the model space
            objfn : callable 
                Takes CI vectors as a kwarg and returns the value,
                gradient, and Hessian of a chosen objective function wrt
                rotation between pairs of CI vectors
            max_cyc : integer
                Maximum number of cycles of the gradient-ascent
                algorithm
            conv_tol : float
                Maximum value of both gradient and step vectors at
                convergence
            sing_tol : float
                Tolerance for determining when normal coordinate belongs
                to the null space (df = d2f = 0) or when the Hessian is
                singular (df != 0, d2f = 0).
            nudge_tol : float
                Minimum step size along a normal coordinate when the
                surface is locally concave.

        Returns:
            conv : logical
                True if the optimization is converged
            ci : list of len (nroots)
                Optimized CI vectors describing intermediate states
    '''

    if ci is None: ci = mc.ci
    if objfn is None: objfn = mc.sarot_objfn
    if max_cyc is None: max_cyc = getattr (mc, 'max_cyc_sarot', 50)
    if conv_tol is None: conv_tol = getattr (mc, 'conv_tol_sarot', 1e-8)
    if sing_tol is None: sing_tol = getattr (mc, 'sing_tol_sarot', 1e-8)
    if nudge_tol is None: nudge_tol = getattr (mc, 'nudge_tol_sarot', 1e-3)
    ci = np.array (ci) # copy
    ci_old = ci.copy ()
    log = lib.logger.new_logger (mc, mc.verbose)
    nroots = mc.fcisolver.nroots 
    rows,col = np.tril_indices(nroots,k=-1)
    npairs = nroots * (nroots - 1) // 2
    t = np.zeros((nroots,nroots))
    conv = False
    hdr = '{} intermediate-state'.format (mc.__class__.__name__)

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

        f, df, d2f = objfn (ci=ci)
        log.info ("{} objective function value = {}".format (hdr, f))

        # Analyze Hessian
        d2f, evecs = linalg.eigh (d2f)
        evecs = np.array(evecs)
        df = np.dot (df, evecs)
        d2f_zero = np.abs (d2f) < sing_tol
        df_zero = np.abs (df) < sing_tol
        if np.any (d2f_zero & (~df_zero)):
            log.warn ("{} Hess is singular!".format (hdr))
        idx_null = d2f_zero & df_zero
        df[idx_null] = 0.0
        d2f[idx_null] = -1e-16
        pos_idx = d2f > 0
        neg_def = np.all (d2f < 0)
        log.info ("{} Hessian is negative-definite? {}".format (hdr, neg_def))

        # Analyze gradient
        grad_norm = np.linalg.norm(df)
        log.info ("{} grad norm = %f".format (hdr), grad_norm)
        log.info ("{} grad (normal modes) = {}".format (hdr, df))

        # Take step
        df[pos_idx & (np.abs (df/d2f) < nudge_tol)] = nudge_tol
        Dt = df/np.abs (d2f)
        step_norm = np.linalg.norm (Dt)
        log.info ("{} Hessian eigenvalues: {}".format (hdr, d2f))
        log.info ("{} step vector (normal modes): {}".format (hdr, Dt))
        t[:] = 0
        t[np.tril_indices(t.shape[0], k = -1)] = np.dot (Dt, evecs.T)
        t = t - t.T

        if grad_norm < conv_tol and step_norm < conv_tol and neg_def == True:
                conv = True
                break

    U_signed = np.tensordot (ci_old, ci.conj (), axes=((1,2),(1,2)))
    if mc.verbose >= lib.logger.DEBUG:
        fmt_str = ' ' + ' '.join (['{:5.2f}',]*nroots)
        log.debug ("{} final overlap matrix:".format (hdr))
        for row in U_signed: log.debug (fmt_str.format (*row))
    # Root order and sign by overlap criterion
    # Requires ~strictly~ non-repeating sort
    # TODO: generalize to only sort within solvers in
    # SA-mix (can probably hack this using U_abs)
    #U_abs = np.abs (U_signed)
    #sgn = np.ones (nroots)
    #ovlp_idx = -sgn.copy ().astype (np.int32)
    #for imax in range (nroots):
    #    i = np.argmax (U_abs)
    #    j, k = i // nroots, i % nroots
    #    sgn[j] -= 2 * int (U_signed[j,k] < 0)
    #    ovlp_idx[j] = k
    #    U_abs[j,:] = -1
    #log.debug ("{} sign-permutation array: {}".format (hdr, sgn))
    #log.debug ("{} overlap sort array: {}".format (hdr, ovlp_idx))
    #ci *= sgn[:,None,None]
    #ci = ci[ovlp_idx,:,:]

    if conv:
        log.note ("{} optimization CONVERGED".format (hdr))
    else:
        log.note (("{} optimization did not converge after {} "
                   "cycles".format (hdr, it)))

    return conv, list (ci)

class StateInteractionMCPDFTSolver ():
    pass
    # tag

# TODO: better docstring
class _SIPDFT (StateInteractionMCPDFTSolver):
    ''' I'm not going to use subclass to distinguish between various SI-PDFT
        types. Instead, I'm going to use three method attributes:

        _sarot_objfn : callable
            Args: ci vectors
            Returns: float, array (nroots), array (nroots,nroots)
            The value, first, and second derivatives of the objective function
            which extrema define the intermediate states. May be used both in
            performing the SI-PDFT energy calculation and in gradients.

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
        keys = set (('sarot_objfn', 'sarot', 'sarot_name',
                     'heff_mcscf', 'heff_pdft', 'hdiag_mcscf', 'hdiag_pdft',
                     'si', 'si_mcscf', 'si_pdft',
                     'max_cycle_sarot', 'conv_tol_sarot'))
        self._sarot_objfn = sarot_objfn
        self._sarot = sarot
        self.max_cycle_sarot = 50
        self.conv_tol_sarot = 1e-8
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

    @property
    def hdiag_mcscf (self):
        return self.heff_mcscf.diagonal ()

    @property
    def heff_pdft (self):
        arr = self.heff_mcscf.copy ()
        idx = np.diag_indices_from (arr)
        arr[idx] = self.hdiag_pdft
        arr.flags['WRITEABLE'] = False
        return arr

    @property
    def si (self):
        return self.si_pdft
    @si.setter
    def si (self, x):
        self.si_pdft = x

    def _init_ci0 (self, ci0, mo_coeff=None):
        ''' On the assumption that ci0 represents states that optimize
            the SI-PDFT objective function, prediagonalize the
            Hamiltonian so that the MC-SCF step has a better
            initialization. 
        '''
        # TODO: different spin states in state-average-mix ?
        if ci0 is None: ci0 = getattr (self, 'ci', None)
        if ci0 is None: return None
        if mo_coeff is None: mo_coeff = self.mo_coeff
        heff_mcscf = get_heff_mcscf (mo_coeff, ci0)
        e, u = self._eig_si (heff_mcscf)
        ci = list (np.tensordot (u.T, np.asarray (ci0), axes=1))
        return ci

    def _init_sarot_ci (self, ci, ci0):
        ''' On the assumption that ci0 represents states that optimize
            the SI-PDFT objective function, rotate the MC-SCF ci vectors
            to maximize their overlap with ci0 so that the sarot step
            has a better initialization.'''
        # TODO: different spin states in state-average-mix ?
        if ci0 is None: return None
        ci0_array = np.asarray (ci0)
        ci_array = np.asarray (ci)
        ovlp = np.tensordot (ci0_array.conj (), ci_array, axes=((1,2),(1,2)))
        u, svals, vh = linalg.svd (ovlp)
        ci = list (np.tensordot (u @ vh, ci_array, axes=1))
        return ci

    # TODO: docstring
    def kernel (self, mo_coeff=None, ci0=None, otxc=None, grids_level=None,
                grids_attr=None, **kwargs):
        # Reference state determination
        self.otfnal.reset (mol=self.mol) # scanner mode safety 
        ci = self._init_ci0 (ci0, mo_coeff=mo_coeff)
        self.optimize_mcscf_(mo_coeff=mo_coeff,ci0=ci)
        # Intermediate state determination
        ci = self._init_sarot_ci (ci, ci0)
        sarot_conv, self.ci = self.sarot (ci=ci, **kwargs)
        self.converged = self.converged and sarot_conv
        # Energy calculation
        self.heff_mcscf = self.make_heff_mcscf ()
        self.hdiag_pdft = self.compute_pdft_energy_(
            otxc=otxc, grids_level=grids_level, grids_attr=grids_attr)[-1]
        # Final diagonalization
        e_mcscf, self.si_mcscf = self._eig_si (self.heff_mcscf)
        if abs (linalg.norm (self.e_mcscf-e_mcscf)) > 1e-10:
            raise RuntimeError (("Sanity fault: e_mcscf ({}) != "
                                "self.e_mcscf ({})").format (e_mcscf,
                                self.e_mcscf))
        self.e_states, self.si_pdft = self._eig_si (self.heff_pdft)
        self.e_tot = np.dot (self.e_states, self.weights)
        self._log_sarot ()
        self._log_si ()
        return (self.e_tot, self.e_ot, self.e_mcscf, self.e_cas, self.ci, 
            self.mo_coeff, self.mo_energy)

    # All of the below probably need to be wrapped over solvers in
    # state-interaction-mix metaclass

    def sarot (self, ci=None, **kwargs):
        ''' Optimize the intermediate states describing the model space
            of an MS-PDFT calculation. The specific algorithm depends on
            the specific MS method; see the docstring for this object's
            _sarot member.
        '''
        if ci is None: ci = self.ci
        return self._sarot (self, ci, **kwargs)

    def sarot_objfn (self, mo_coeff=None, ci=None):
        ''' Computes the value, gradient vector, and Hessian matrix of
            the objective function rendered stationary by the
            optimized intermediate states. Used in gradient calculations
            and possibly in sarot. The details depend on the specific MS
            method; see the docstring for this object's _sarot_objfn
            member.
        '''
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ci is None: ci = self.ci
        return self._sarot_objfn (self, mo_coeff=mo_coeff, ci=ci)

    def _eig_si (self, heff):
        return linalg.eigh (heff)

    make_heff_mcscf = make_heff_mcscf

    def _log_sarot (self):
        # Information about the intermediate states
        hdiag_mcscf = self.hdiag_mcscf
        hdiag_pdft = self.hdiag_pdft
        nroots = len (hdiag_pdft)
        log = lib.logger.new_logger (self, self.verbose)
        f, df, d2f = self.sarot_objfn ()
        hdr = '{} intermediate'.format (self.__class__.__name__)
        log.note ('%s objective function  value = %.15g |grad| = %.7g', hdr, f, linalg.norm (df))
        log.note ('%s average energy  EPDFT = %.15g  EMCSCF = %.15g', hdr,
                  np.dot (self.weights, hdiag_pdft), np.dot (self.weights, hdiag_mcscf))
        log.note ('%s states:', hdr)
        if getattr (self.fcisolver, 'spin_square', None):
            ss = self.fcisolver.states_spin_square (self.ci, self.ncas,
                                                    self.nelecas)[0]
            for i in range (nroots):
                log.note ('  State %d weight %g  EPDFT = %.15g  EMCSCF = %.15g'
                    '  S^2 = %.7f', i, self.weights[i], hdiag_pdft[i],
                    hdiag_mcscf[i], ss[i])
        else:
            for i in range (nroots):
                log.note ('  State %d weight %g  EPDFT = %.15g  EMCSCF = '
                    '%.15g', i, self.weights[i], hdiag_pdft[i], hdiag_mcscf[i])
        log.info ('Intermediate state MS-PDFT effective Hamiltonian matrix:')
        fmt_str = ' '.join (['{:9.5f}',]*nroots)
        for row in self.heff_pdft: log.info (fmt_str.format (*row))
        log.info ('Intermediate states (columns) in terms of reference states '
            '(rows):')
        for row in self.si_mcscf.T: log.info (fmt_str.format (*row))

    def _log_si (self):
        # Information about the final states
        log = lib.logger.new_logger (self, self.verbose)
        nroots = len (self.e_states)
        log.note ('%s final states:', self.__class__.__name__) 
        if getattr (self.fcisolver, 'spin_square', None):
            ci = np.tensordot (self.si.T, np.asarray (self.ci), axes=1)
            ss = self.fcisolver.states_spin_square (ci, self.ncas,
                                                    self.nelecas)[0]
            for i in range (nroots):
                log.note ('  State %d weight %g  EMSPDFT = %.15g  S^2 = %.7f',
                          i, self.weights[i], self.e_states[i], ss[i])
        else:
            for i in range (nroots):
                log.note ('  State %d weight %g  EMSPDFT = %.15g', i,
                          self.weights[i], self.e_states[i])

    def nuc_grad_method (self):
        from mrh.my_pyscf.grad.sipdft import Gradients
        return Gradients (self)

def get_sarotfns (obj):
    ''' Interpret the name of the MS-PDFT method as a pair of functions
        which optimize the intermediate states and calculate the power
        series in the corresponding objective function to second order.

        Args:
            obj : string
                Specify particular MS-PDFT method. Currently, only "CMS"
                is supported. Not case-sensitive.

        Returns:
            sarot_objfn : callable
                Takes model-space CI vectors in a trial intermediate-
                state basis and returns the value and first and second
                derivatives of the objective function specified by obj
            sarot : callable
                Takes model-space CI vectors and returns CI vectors in
                the optimized intermediate-state basis
    '''

    if obj.upper () == 'CMS':
        from mrh.my_pyscf.mcpdft.cmspdft import e_coul as sarot_objfn
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
        base_name = mc.__class__.__name__
        mc = mc.state_average (weights=weights, **kwargs)
    else:
        base_name = mc.__class__.__bases__[0].__name__
    mcbase_class = mc.__class__
    sarot_objfn, sarot = get_sarotfns (obj)

    class SIPDFT (_SIPDFT, mcbase_class):
        pass
    SIPDFT.__name__ = obj.upper () + base_name
    return SIPDFT (mc, sarot_objfn, sarot, obj)
    

if __name__ == '__main__':
    # This ^ is a convenient way to debug code that you are working on. The
    # code in this block will only execute if you run this python script as the
    # input directly: "python sipdft.py".

    from pyscf import scf, gto
    from mrh.my_pyscf.tools import molden # My version is better for MC-SCF
    from mrh.my_pyscf.fci import csf_solver
    xyz = '''O  0.00000000   0.08111156   0.00000000
             H  0.78620605   0.66349738   0.00000000
             H -0.78620605   0.66349738   0.00000000'''
    mol = gto.M (atom=xyz, basis='sto-3g', symmetry=False, output='sipdft.log',
        verbose=lib.logger.DEBUG)
    mf = scf.RHF (mol).run ()
    mc = mcpdft.CASSCF (mf, 'tPBE', 4, 4).set (fcisolver = csf_solver (mol, 1))
    mc = mc.state_interaction ([1.0/3,]*3, 'cms').run ()


