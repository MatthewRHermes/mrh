import time
from pyscf import lib, __config__
from pyscf.grad import rhf as rhf_grad
import numpy as np
from scipy import linalg

default_level_shift = getattr(__config__, 'mcscf_mc1step_CASSCF_ah_level_shift', 1e-8)
default_conv_tol = getattr (__config__, 'mcscf_mc1step_CASSCF_ah_conv_tol', 1e-12)
default_max_cycle = getattr (__config__, 'mcscf_mc1step_CASSCF_max_cycle', 50)
default_lindep = getattr (__config__, 'mcscf_mc1step_CASSCF_lindep', 1e-14)

class Gradients (lib.StreamObject):
''' Dummy parent class for calculating analytical nuclear gradients using the technique of Lagrange multipliers:
    L = E + \sum_i z_i L_i
    dE/dx = \partial L/\partial x iff all L_i = 0 for the given wave function
    I.E., the Lagrange multipliers L_i cancel the direct dependence of the wave function on the nuclear coordinates
    and allow the Hellmann-Feynman theorem to be used for some non-variational methods. '''

    ################################## Child classes MUST overwrite the methods below ################################################

    def get_wfn_response (self, **kwargs):
        ''' Return first derivative of the energy wrt wave function parameters conjugate to the Lagrange multipliers.
            Used to calculate the value of the Lagrange multipliers. '''
        return np.zeros (nlag)

    def get_Lop_Ldiag (self, **kwargs):
        ''' Return a function calculating Lvec . J_wfn, where J_wfn is the Jacobian of the Lagrange cofactors (e.g.,
            in state-averaged CASSCF, the Hessian of the state-averaged energy wrt wfn parameters) along with
            the diagonal of the Jacobian. '''
        def Lop (Lvec):
            return np.zeros (nlag)
        Ldiag = np.zeros (nlag)
        return Lop, Ldiag
    
    def get_nuc_response (self, **kwargs):
        ''' Return expectation values <dH/dx> where x is nuclear displacement. I.E., the gradient if the method were variational. '''
        return np.zeros ((natm, 3))

    def get_LdotJnuc (self, Lvec, **kwargs):
        ''' Return Lvec . J_nuc, where J_nuc is the Jacobian of the Lagrange cofactors wrt nuclear displacement. This is the
            second term of the final gradient expectation value. '''
        return np.zeros ((natm, 3))

    ################################## Child classes SHOULD overwrite the methods below ##############################################

    def __init__(self, mol, nlag, method):
        self.mol = mol
        self.base = method
        self.verbose = mol.verbose
        self.stdout = mol.stdout
        self.nlag = nlag
        self.natm = mol.natm
        self.atmlst = None
        self.de = None
        self._keys = set (self.__dict__.keys ())
        #--------------------------------------#
        self.level_shift = default_level_shift
        self.conv_tol = default_conv_tol
        self.max_cycle = default_max_cycle
        self.lindep = default_lindep

    def get_lagrange_precond (self, Ldiag, level_shift=None):
        ''' Default preconditioner for solving for the Lagrange multipliers: 1/(Ldiag-shift) '''
        if level_shift is None: level_shift = self.level_shift
        def my_precond (x, e):
            Ldiagd = Ldiag - (e * level_shift)
            Ldiagd[abs(Ldiagd)<1e-8] = 1e-8
            x /= Ldiagd
            x /= linalg.norm (x)
            return x
        return my_precond

    ################################## Child classes SHOULD NOT overwrite the methods below ###########################################

    def solve_lagrange (self, Lvec_guess=None):
        rvec = self.get_wfn_response ()
        Lop, Ldiag = self.get_Lop_Ldiag ()
        precond = self.get_lagrange_precond (Ldiag, level_shift=self.level_shift)
        rvec_op = lambda *args: rvec
        log = lib.logger.new_logger (self, self.verbose)
        if Lvec_guess is None: Lvec_guess = rvec
        for conv, ihop, eig, Lvec, LdotJ, residual, seig \
                in ciah.davidson_cc(Lop, rvec_op, precond, Lvec_guess,
                                tol=self.conv_tol, max_cycle=self.max_cycle,
                                lindep=self.lindep, verbose=log):
            norm_geff = linalg.norm (rvec + LdotJ)
            norm_Lvec = linalg.norm (Lvec)
            log.debug('    iter %d  |rvec+LdotJ|=%3.2e |Lvec|=%3.2e eig=%2.1e seig=%2.1e',
                      ihop, norm_geff, norm_Lvec, w, seig)
            if conv or ihop >= max_cycle:
                break
        if conv:
            log.info ('Lagrange multipliers converged to %8.4e after %d iterations', conv_tol, ihop)
        else:
            log.info ('Lagrange multiplier determination failed to converge to %8.4e after '
                '%d iterations (residual norm: %8.4e; Lvec norm: %8.4e)', conv_tol, ihop, norm_geff, norm_Lvec)
        return conv, Lvec
                    
    def kernel (self, **kwargs):
        cput0 = (time.clock(), time.time())
        log = logger.new_logger(self, verbose)
        if 'atmlst' in kwargs:
            self.atmlst = kwargs['atmlst']
        self.natm = len (self.atmlst)

        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose >= logger.INFO:
            self.dump_flags()

        conv, Lvec = self.solve_lagrange (**kwargs)
        assert (conv), 'Lagrange convergence failure'

        self.de = self.get_wfn_response (**kwargs) + self.get_LdotJnuc (Lvec, **kwargs)
        log.timer('Lagrange gradients', *cput0)
        self._finalize()
        return self.de

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('\n')
        if not self.base.converged:
            log.warn('Ground state method not converged')
        log.info('******** %s for %s ********',
                 self.__class__, self.base.__class__)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        return self

    def _finalize (self,
        if self.verbose >= lib.logger.NOTE:
            lib.logger.note(self, '--------------- %s gradients ---------------',
                        self.base.__class__.__name__)
            rhf_grad._write(self, self.mol, self.de, self.atmlst)
            logger.note(self, '----------------------------------------------')


