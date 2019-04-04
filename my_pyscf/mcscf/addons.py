#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import sys
import copy
from functools import reduce
import numpy
from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
from pyscf import fci
from pyscf import scf, mcscf
from pyscf import symm
from pyscf import __config__
from pyscf.mcscf.addons import StateAverageFCISolver
from mrh.my_pyscf.mcscf import newton_sacasscf, newton_sacasscf_symm

# MRH: I prefer to refer to numpy as 'np'
np = numpy

BASE = getattr(__config__, 'mcscf_addons_sort_mo_base', 1)
MAP2HF_TOL = getattr(__config__, 'mcscf_addons_map2hf_tol', 0.4)

def state_average_(casscf, weights=(0.5,0.5)):
    ''' State average over the energy.  The energy funcitonal is
    E = w1<psi1|H|psi1> + w2<psi2|H|psi2> + ...

    Note we may need change the FCI solver to

    mc.fcisolver = fci.solver(mol, False)

    before calling state_average_(mc), to mix the singlet and triplet states
    '''
    # MRH: weights as a np array because this will be important later
    weights = np.asarray (weights)
    assert(abs(sum(weights)-1) < 1e-3)
    fcibase_class = casscf.fcisolver.__class__
    if fcibase_class.__name__ == 'FakeCISolver':
        raise TypeError('mc.fcisolver is not base FCI solver\n'
                        'state_average function cannot work with decorated '
                        'fcisolver %s.\nYou can restore the base fcisolver '
                        'then call state_average function, eg\n'
                        '    mc.fcisolver = %s.%s(mc.mol)\n'
                        '    mc.state_average_()\n' %
                        (casscf.fcisolver, fcibase_class.__base__.__module__,
                         fcibase_class.__base__.__name__))
    has_spin_square = getattr(casscf.fcisolver, 'spin_square', None)

    # MRH: turn weights into a member. I can deal with the _keys BS later
    class FakeCISolver(fcibase_class, StateAverageFCISolver):
        def __init__(self, mol=None):
            self.nroots = len(weights)
            self.weights = weights
            self.e_states = [None]
            self.fcibase_class = fcibase_class
        def kernel(self, h1, h2, norb, nelec, ci0=None, **kwargs):
# pass self to fcibase_class.kernel function because orbsym argument is stored in self
# but undefined in fcibase object
            e, c = fcibase_class.kernel(self, h1, h2, norb, nelec, ci0,
                                        nroots=self.nroots, **kwargs)
            self.e_states[0] = e
            if casscf.verbose >= logger.DEBUG:
                if has_spin_square:
                    for i, ei in enumerate(e):
                        ss = fcibase_class.spin_square(self, c[i], norb, nelec)
                        logger.debug(casscf, 'state %d  E = %.15g S^2 = %.7f',
                                     i, ei, ss[0])
                else:
                    for i, ei in enumerate(e):
                        logger.debug(casscf, 'state %d  E = %.15g', i, ei)
            return numpy.einsum('i,i->', e, self.weights), c
        def approx_kernel(self, h1, h2, norb, nelec, ci0=None, **kwargs):
            e, c = fcibase_class.kernel(self, h1, h2, norb, nelec, ci0,
                                        max_cycle=casscf.ci_response_space,
                                        nroots=self.nroots, **kwargs)
            return numpy.einsum('i,i->', e, self.weights), c
        def make_rdm1(self, ci0, norb, nelec):
            dm1 = 0
            for i, wi in enumerate(self.weights):
                dm1 += wi * fcibase_class.make_rdm1(self, ci0[i], norb, nelec)
            return dm1
        def make_rdm1s(self, ci0, norb, nelec):
            dm1a, dm1b = 0, 0
            for i, wi in enumerate(self.weights):
                dm1s = fcibase_class.make_rdm1s(self, ci0[i], norb, nelec)
                dm1a += wi * dm1s[0]
                dm1b += wi * dm1s[1]
            return dm1a, dm1b
        def make_rdm12(self, ci0, norb, nelec, link_index=None):
            rdm1 = 0
            rdm2 = 0
            for i, wi in enumerate(self.weights):
                dm1, dm2 = fcibase_class.make_rdm12(self, ci0[i], norb, nelec, link_index=link_index)
                rdm1 += wi * dm1
                rdm2 += wi * dm2
            return rdm1, rdm2

        if has_spin_square:
            def spin_square(self, ci0, norb, nelec):
                ss = 0
                multip = 0
                for i, wi in enumerate(self.weights):
                    res = fcibase_class.spin_square(self, ci0[i], norb, nelec)
                    ss += wi * res[0]
                    multip += wi * res[1]
                return ss, multip

    fcisolver = FakeCISolver(casscf.mol)
    fcisolver.__dict__.update(casscf.fcisolver.__dict__)
    fcisolver.nroots = len(weights)
    casscf.fcisolver = fcisolver

    # MRH: Here is where I intervene in my quest to make SA-MCPDFT gradients
    # First, I need the Newton-algorithm CASSCF to work. I also need to make the weights available to the
    # CASSCF object.

    class SACASSCF(casscf.__class__):

        def __init__(self, my_mc, my_weights):
            self.__dict__.update (my_mc.__dict__)
            self.fcisolver.weights = my_weights
            self.fcisolver = fcisolver

        def _finalize(self):
            # MRH: this is how it should have been done from the beginning!!!!
            super()._finalize ()
            self.e_tot = self.fcisolver.e_states[0]
            logger.note(self, 'CASCI energy for each state')
            if has_spin_square:
                ncas = self.ncas
                nelecas = self.nelecas
                for i, ei in enumerate(self.e_tot):
                    ss = fcibase_class.spin_square(self.fcisolver, self.ci[i],
                                                   ncas, nelecas)[0]
                    logger.note(self, '  State %d weight %g  E = %.15g S^2 = %.7f',
                                i, self.weights[i], ei, ss)
            else:
                for i, ei in enumerate(self.e_tot):
                    logger.note(self, '  State %d weight %g  E = %.15g',
                                i, self.weights[i], ei)
            return self

        @property
        def weights (self):
            return self.fcisolver.weights

        @weights.setter
        def weights (self, x):
            self.fcisolver.weights = x

        def newton (self):
            if isinstance (self, mcscf.casci_symm.SymAdaptedCASCI):
                mc1 = newton_sacasscf_symm.CASSCF (self._scf, self.ncas, self.nelecas)
            else:
                mc1 = newton_sacasscf.CASSCF (self._scf, self.ncas, self.nelecas)
            mc1.__dict__.update(self.__dict__)
            mc1.max_cycle_micro = 10
            mc1.fcisolver = fcisolver
            return mc1

    sa_mc = SACASSCF (casscf, weights)
    if isinstance (casscf, (mcscf.newton_casscf.CASSCF)):
        return sa_mc.newton ()
    return sa_mc

state_average = state_average_

'''
def state_specific_(casscf, state=1):
    For excited state

    Kwargs:
        state : int
        0 for ground state; 1 for first excited state.
    
    fcibase_class = casscf.fcisolver.__class__
    if fcibase_class.__name__ == 'FakeCISolver':
        raise TypeError('mc.fcisolver is not base FCI solver\n'
                        'state_specific function cannot work with decorated '
                        'fcisolver %s.\nYou can restore the base fcisolver '
                        'then call state_specific function, eg\n'
                        '    mc.fcisolver = %s.%s(mc.mol)\n'
                        '    mc.state_specific_()\n' %
                        (casscf.fcisolver, fcibase_class.__base__.__module__,
                         fcibase_class.__base__.__name__))
    class FakeCISolver(fcibase_class, StateAverageFCISolver):
        def __init__(self):
            self.nroots = state+1
            self._civec = None
        def kernel(self, h1, h2, norb, nelec, ci0=None, **kwargs):
            if self._civec is not None:
                ci0 = self._civec
            e, c = fcibase_class.kernel(self, h1, h2, norb, nelec, ci0,
                                        nroots=self.nroots, **kwargs)
            if state == 0:
                e = [e]
                c = [c]
            self._civec = c
            if casscf.verbose >= logger.DEBUG:
                if getattr(fcibase_class, 'spin_square', None):
                    ss = fcibase_class.spin_square(self, c[state], norb, nelec)
                    logger.debug(casscf, 'state %d  E = %.15g S^2 = %.7f',
                                 state, e[state], ss[0])
                else:
                    logger.debug(casscf, 'state %d  E = %.15g', state, e[state])
            return e[state], c[state]
        def approx_kernel(self, h1, h2, norb, nelec, ci0=None, **kwargs):
            if self._civec is not None:
                ci0 = self._civec
            e, c = fcibase_class.kernel(self, h1, h2, norb, nelec, ci0,
                                        max_cycle=casscf.ci_response_space,
                                        nroots=self.nroots, **kwargs)
            if state == 0:
                self._civec = [c]
                return e, c
            else:
                self._civec = c
                return e[state], c[state]

    fcisolver = FakeCISolver()
    fcisolver.__dict__.update(casscf.fcisolver.__dict__)
    fcisolver.nroots = state+1
    casscf.fcisolver = fcisolver
    return casscf
state_specific = state_specific_

def state_average_mix_(casscf, fcisolvers, weights=(0.5,0.5)):
    State-average CASSCF over multiple FCI solvers.
    
    fcibase_class = fcisolvers[0].__class__
#    if fcibase_class.__name__ == 'FakeCISolver':
#        logger.warn(casscf, 'casscf.fcisolver %s is a decorated FCI solver. '
#                    'state_average_mix_ function rolls back to the base solver %s',
#                    fcibase_class, fcibase_class.__base__)
#        fcibase_class = fcibase_class.__base__
    nroots = sum(solver.nroots for solver in fcisolvers)
    assert(nroots == len(weights))
    has_spin_square = all(getattr(solver, 'spin_square', None)
                          for solver in fcisolvers)
    e_states = [None]

    def collect(items):
        items = list(items)
        cols = [[item[i] for item in items] for i in range(len(items[0]))]
        return cols
    def loop_solver(solvers, ci0):
        p0 = 0
        for solver in solvers:
            if ci0 is None:
                yield solver, None
            elif solver.nroots == 1:
                yield solver, ci0[p0]
            else:
                yield solver, ci0[p0:p0+solver.nroots]
            p0 += solver.nroots
    def loop_civecs(solvers, ci0):
        p0 = 0
        for solver in solvers:
            for i in range(p0, p0+solver.nroots):
                yield solver, ci0[i]
            p0 += solver.nroots
    def get_nelec(solver, nelec):
        # FCISolver does not need this function. Some external solver may not
        # have the function to handle nelec and spin
        if solver.spin is not None:
            nelec = numpy.sum(nelec)
            nelec = (nelec+solver.spin)//2, (nelec-solver.spin)//2
        return nelec

    class FakeCISolver(fcibase_class, StateAverageFCISolver):
        def kernel(self, h1, h2, norb, nelec, ci0=None, verbose=0, **kwargs):
# Note self.orbsym is initialized lazily in mc1step_symm.kernel function
            log = logger.new_logger(self, verbose)
            es = []
            cs = []
            for solver, c0 in loop_solver(fcisolvers, ci0):
                e, c = solver.kernel(h1, h2, norb, get_nelec(solver, nelec), c0,
                                     orbsym=self.orbsym, verbose=log, **kwargs)
                if solver.nroots == 1:
                    es.append(e)
                    cs.append(c)
                else:
                    es.extend(e)
                    cs.extend(c)
            e_states[0] = es

            if log.verbose >= logger.DEBUG:
                if has_spin_square:
                    ss, multip = collect(solver.spin_square(c0, norb, get_nelec(solver, nelec))
                                         for solver, c0 in loop_civecs(fcisolvers, cs))
                    for i, ei in enumerate(es):
                        log.debug('state %d  E = %.15g S^2 = %.7f', i, ei, ss[i])
                else:
                    for i, ei in enumerate(es):
                        log.debug('state %d  E = %.15g', i, ei)
            return numpy.einsum('i,i', numpy.array(es), weights), cs

        def approx_kernel(self, h1, h2, norb, nelec, ci0=None, **kwargs):
            es = []
            cs = []
            for solver, c0 in loop_solver(fcisolvers, ci0):
                e, c = solver.kernel(h1, h2, norb, get_nelec(solver, nelec), c0,
                                     orbsym=self.orbsym, **kwargs)
                if solver.nroots == 1:
                    es.append(e)
                    cs.append(c)
                else:
                    es.extend(e)
                    cs.extend(c)
            return numpy.einsum('i,i->', es, weights), cs
        def make_rdm1(self, ci0, norb, nelec, **kwargs):
            dm1 = 0
            for i, (solver, c) in enumerate(loop_civecs(fcisolvers, ci0)):
                dm1 += weights[i]*solver.make_rdm1(c, norb, get_nelec(solver, nelec), **kwargs)
            return dm1
        def make_rdm1s(self, ci0, norb, nelec, **kwargs):
            dm1a, dm1b = 0, 0
            for i, (solver, c) in enumerate(loop_civecs(fcisolvers, ci0)):
                dm1s = solver.make_rdm1s(c, norb, get_nelec(solver, nelec), **kwargs)
                dm1a += weights[i] * dm1s[0]
                dm1b += weights[i] * dm1s[1]
            return dm1a, dm1b
        def make_rdm12(self, ci0, norb, nelec, **kwargs):
            rdm1 = 0
            rdm2 = 0
            for i, (solver, c) in enumerate(loop_civecs(fcisolvers, ci0)):
                dm1, dm2 = solver.make_rdm12(c, norb, get_nelec(solver, nelec), **kwargs)
                rdm1 += weights[i] * dm1
                rdm2 += weights[i] * dm2
            return rdm1, rdm2

        if has_spin_square:
            def spin_square(self, ci0, norb, nelec):
                ss = 0
                multip = 0
                for i, (solver, c) in enumerate(loop_civecs(fcisolvers, ci0)):
                    res = solver.spin_square(c, norb, nelec)
                    ss += weights[i] * res[0]
                    multip += weights[i] * res[1]
                return ss, multip

    fcisolver = FakeCISolver(casscf.mol)
    fcisolver.__dict__.update(casscf.fcisolver.__dict__)
    fcisolver.fcisolvers = fcisolvers
    casscf.fcisolver = fcisolver

    old_finalize = casscf._finalize
    def _finalize():
        old_finalize()
        casscf.e_tot = e_states[0]
        logger.note(casscf, 'CASCI energy for each state')
        if has_spin_square:
            ncas = casscf.ncas
            nelecas = casscf.nelecas
            ss, multip = collect(solver.spin_square(c0, ncas, get_nelec(solver, nelecas))
                                 for solver, c0 in loop_civecs(fcisolvers, casscf.ci))
            for i, ei in enumerate(casscf.e_tot):
                logger.note(casscf, '  State %d weight %g  E = %.15g S^2 = %.7f',
                            i, weights[i], ei, ss[i])
        else:
            for i, ei in enumerate(casscf.e_tot):
                logger.note(casscf, '  State %d weight %g  E = %.15g',
                            i, weights[i], ei)
        return casscf
    casscf._finalize = _finalize
    return casscf
state_average_mix = state_average_mix_

del(BASE, MAP2HF_TOL)
'''

