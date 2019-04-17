import time
from pyscf import lib, __config__
from pyscf.grad import rhf as rhf_grad
from pyscf.soscf import ciah
import numpy as np
from scipy import linalg

default_level_shift = getattr(__config__, 'mcscf_mc1step_CASSCF_ah_level_shift', 1e-8)
default_conv_tol = getattr (__config__, 'mcscf_mc1step_CASSCF_ah_conv_tol', 1e-12)
default_max_cycle = getattr (__config__, 'mcscf_mc1step_CASSCF_max_cycle', 50) * 3
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

    def get_Aop_Adiag (self, **kwargs):
        ''' Return a function calculating Lvec . J_wfn, where J_wfn is the Jacobian of the Lagrange cofactors (e.g.,
            in state-averaged CASSCF, the Hessian of the state-averaged energy wrt wfn parameters) along with
            the diagonal of the Jacobian. '''
        def Aop (Lvec):
            return np.zeros (nlag)
        Adiag = np.zeros (nlag)
        return Aop, Adiag
    
    def get_ham_response (self, **kwargs):
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
        self.atmlst = list (range (self.natm))
        self.de = None
        self._keys = set (self.__dict__.keys ())
        #--------------------------------------#
        self.level_shift = default_level_shift
        self.conv_tol = default_conv_tol
        self.max_cycle = default_max_cycle
        self.lindep = default_lindep

    def debug_lagrange (self, Lvec, bvec, Aop, Adiag, **kwargs):
        lib.logger.debug (self, "{} gradient Lagrange factor debugging not enabled".format (self.base.__class__.__name__))
        pass

    def get_lagrange_callback (self, Lvec_last, itvec, geff_op):
        def my_call (x, e):
            itvec[0] += 1
            lib.logger.debug (self, 'Lagrange optimization iteration {}, |geff| = {}, |dLvec| = {}'.format (itvec[0],
                linalg.norm (geff_op (x)), linalg.norm (x - Lvec_last))) 
            Lvec_last[:] = x[:]
        return my_call

    def get_lagrange_precond (self, bvec, Adiag, Aop, Lvec_op=None, geff_op=None, level_shift=None):
        ''' Default preconditioner for solving for the Lagrange multipliers: 1/(Adiag-shift) '''
        if level_shift is None: level_shift = self.level_shift
        def my_precond (x, e):
            e = (x * (bvec + Aop (x))).sum () 
            Adiagd = Adiag - e + level_shift
            Adiagd[abs(Adiagd)<1e-8] = 1e-8
            x /= Adiagd
            return x
        return my_precond


    ################################## Child classes SHOULD NOT overwrite the methods below ###########################################

    def solve_lagrange (self, Lvec_guess=None, **kwargs):
        bvec = self.get_wfn_response (**kwargs)
        Aop, Adiag = self.get_Aop_Adiag (**kwargs)
        Lvec = np.zeros_like (bvec)
        def Lvec_op ():
            return Lvec
        geff = bvec.copy ()
        geff_op = lambda *args: geff
        precond = self.get_lagrange_precond (geff, Adiag, Aop, Lvec_op=Lvec_op, geff_op=geff_op, level_shift=self.level_shift, **kwargs)
        log = lib.logger.new_logger (self, self.verbose)
        if Lvec_guess is None: Lvec_guess = geff
        for conv, ihop, eig, dLvec, dgeff, residual, seig \
                in ciah.davidson_cc(Aop, geff_op, precond, Lvec_guess,
                                tol=self.conv_tol, max_cycle=self.max_cycle,
                                lindep=self.lindep, verbose=log):
            Lvec += dLvec
            geff += dgeff
            log.debug('    iter %d  |geff|=%3.2e |dLvec|=%3.2e eig=%2.1e seig=%2.1e',
                      ihop, linalg.norm (geff), linalg.norm (dLvec), eig, seig)
            if conv or ihop >= self.max_cycle:
                break
        if conv:
            log.info ('Lagrange multipliers converged to %8.4e after %d iterations', self.conv_tol, ihop)
        else:
            log.info ('Lagrange multiplier determination failed to converge to %8.4e after '
                '%d iterations (residual norm: %8.4e; Lvec norm: %8.4e)', self.conv_tol, ihop, linalg.norm (geff), linalg.norm (Lvec))
        return conv, Lvec, bvec, Aop, Adiag
                    
    def kernel (self, **kwargs):
        cput0 = (time.clock(), time.time())
        log = lib.logger.new_logger(self, self.verbose)
        if 'atmlst' in kwargs:
            self.atmlst = kwargs['atmlst']
        self.natm = len (self.atmlst)

        if self.verbose >= lib.logger.WARN:
            self.check_sanity()
        if self.verbose >= lib.logger.INFO:
            self.dump_flags()

        conv, Lvec, bvec, Aop, Adiag = self.solve_lagrange (**kwargs)
        self.debug_lagrange (Lvec, bvec, Aop, Adiag, **kwargs)

        ham_response = self.get_ham_response (**kwargs)
        lib.logger.info(self, '--------------- %s gradient Hamiltonian response ---------------',
                    self.base.__class__.__name__)
        rhf_grad._write(self, self.mol, ham_response, self.atmlst)
        lib.logger.info(self, '----------------------------------------------')

        LdotJnuc = self.get_LdotJnuc (Lvec, **kwargs)
        lib.logger.info(self, '--------------- %s gradient Lagrange response ---------------',
                    self.base.__class__.__name__)
        rhf_grad._write(self, self.mol, LdotJnuc, self.atmlst)
        lib.logger.info(self, '----------------------------------------------')
        
        self.de = ham_response + LdotJnuc
        log.timer('Lagrange gradients', *cput0)
        self._finalize()
        return self.de

    def dump_flags(self):
        log = lib.logger.Logger(self.stdout, self.verbose)
        log.info('\n')
        if not self.base.converged:
            log.warn('Ground state method not converged')
        log.info('******** %s for %s ********',
                 self.__class__, self.base.__class__)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        return self

    def _finalize (self):
        if self.verbose >= lib.logger.NOTE:
            lib.logger.note(self, '--------------- %s gradients ---------------',
                        self.base.__class__.__name__)
            rhf_grad._write(self, self.mol, self.de, self.atmlst)
            lib.logger.note(self, '----------------------------------------------')


