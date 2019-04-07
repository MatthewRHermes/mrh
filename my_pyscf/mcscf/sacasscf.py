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
from mrh.my_pyscf.mcscf import newton_sacasscf
from pyscf.mcscf import newton_casscf, newton_casscf_symm

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
    is_symm = isinstance (casscf, (mcscf.mc1step_symm.CASSCF, newton_casscf_symm.CASSCF))
    is_newton = isinstance (casscf, newton_casscf.CASSCF)
    if is_symm:
        sa_mc = SymAdaptedSACASSCF (casscf, weights)
    else:
        sa_mc = SACASSCF (casscf, weights)
    if is_newton: sa_mc = sa_mc.newton ()
    return sa_mc

state_average = state_average_

class SACASSCF(mcscf.mc1step.CASSCF):

    def __init__(self, my_mc, my_weights):
        self.__dict__.update (my_mc.__dict__)
        self.make_FakeCISolver (self.fcisolver, my_weights)
        self._keys = set (self.__dict__.keys ())

    def make_FakeCISolver (self, realsolver, weights):
        fcibase_class = realsolver.__class__
        self.ss_fcisolver = realsolver
        has_spin_square = getattr(realsolver, 'spin_square', None)
        class FakeCISolver(fcibase_class, StateAverageFCISolver):
            def __init__(my_self, my_realsolver, my_weights, mol=None):
                my_self.__dict__.update (my_realsolver.__dict__)
                my_self.nroots = len(my_weights)
                my_self.weights = my_weights
                my_self.e_states = [None]
                my_self.fcibase_class = fcibase_class
                my_self.has_spin_square = has_spin_square
                my_self._keys = set (my_self.__dict__)
            def kernel(my_self, h1, h2, norb, nelec, ci0=None, **kwargs):
                # pass self to fcibase_class.kernel function because orbsym argument is stored in self
                # but undefined in fcibase object
                e, c = my_self.fcibase_class.kernel(my_self, h1, h2, norb, nelec, ci0,
                                            nroots=my_self.nroots, **kwargs)
                my_self.e_states[0] = e
                if self.verbose >= logger.DEBUG:
                    if my_self.has_spin_square:
                        for i, ei in enumerate(e):
                            ss = my_self.fcibase_class.spin_square(my_self, c[i], norb, nelec)
                            logger.debug(self, 'state %d  E = %.15g S^2 = %.7f',
                                         i, ei, ss[0])
                    else:
                        for i, ei in enumerate(e):
                            logger.debug(self, 'state %d  E = %.15g', i, ei)
                return numpy.einsum('i,i->', e, my_self.weights), c
            def approx_kernel(my_self, h1, h2, norb, nelec, ci0=None, **kwargs):
                e, c = my_self.fcibase_class.kernel(my_self, h1, h2, norb, nelec, ci0,
                                            max_cycle=self.ci_response_space,
                                            nroots=my_self.nroots, **kwargs)
                return numpy.einsum('i,i->', e, my_self.weights), c
            def make_rdm1(my_self, ci0, norb, nelec):
                dm1 = 0
                for i, wi in enumerate(my_self.weights):
                    dm1 += wi * my_self.fcibase_class.make_rdm1(my_self, ci0[i], norb, nelec)
                return dm1
            def make_rdm1s(my_self, ci0, norb, nelec):
                dm1a, dm1b = 0, 0
                for i, wi in enumerate(my_self.weights):
                    dm1s = my_self.fcibase_class.make_rdm1s(my_self, ci0[i], norb, nelec)
                    dm1a += wi * dm1s[0]
                    dm1b += wi * dm1s[1]
                return dm1a, dm1b
            def make_rdm12(my_self, ci0, norb, nelec, link_index=None):
                rdm1 = 0
                rdm2 = 0
                for i, wi in enumerate(my_self.weights):
                    dm1, dm2 = my_self.fcibase_class.make_rdm12(my_self, ci0[i], norb, nelec, link_index=link_index)
                    rdm1 += wi * dm1
                    rdm2 += wi * dm2
                return rdm1, rdm2

            if has_spin_square:
                def spin_square(my_self, ci0, norb, nelec):
                    ss = 0
                    multip = 0
                    for i, wi in enumerate(my_self.weights):
                        res = my_self.fcibase_class.spin_square(my_self, ci0[i], norb, nelec)
                        ss += wi * res[0]
                        multip += wi * res[1]
                    return ss, multip
        self.fcisolver = FakeCISolver (realsolver, weights, self.mol)

    def _finalize(self):
        # MRH: this is how it should have been done from the beginning!!!!
        super()._finalize ()
        self.e_tot = self.fcisolver.e_states[0]
        logger.note(self, 'CASCI energy for each state')
        if self.fcisolver.has_spin_square:
            ncas = self.ncas
            nelecas = self.nelecas
            for i, ei in enumerate(self.e_tot):
                ss = self.fcisolver.fcibase_class.spin_square(self.fcisolver, self.ci[i],
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
        mc1 = Newton_SACASSCF (self)
        return mc1

class SymAdaptedSACASSCF (mcscf.mc1step_symm.CASSCF, SACASSCF):

    def __init__(self, my_mc, my_weights):
        self.__dict__.update (my_mc.__dict__)
        self.make_FakeCISolver (self.fcisolver, my_weights)
        self._keys = set (self.__dict__.keys ())

    _finalize = SACASSCF._finalize

    def newton (self):
        mc1 = Newton_SymAdaptedSACASSCF (self)
        return mc1

class Newton_SACASSCF(newton_casscf.CASSCF, SACASSCF):

    def __init__(self, my_sacas):
        self.__dict__.update (my_sacas.__dict__)

    def kernel(self, mo_coeff=None, ci0=None, callback=None):
        return mcscf.mc1step.CASSCF.kernel(self, mo_coeff, ci0, callback, newton_sacasscf.kernel)

    _finalize = SACASSCF._finalize

class Newton_SymAdaptedSACASSCF (newton_casscf_symm.CASSCF, SymAdaptedSACASSCF):

    def __init__(self, my_sacas):
        self.__dict__.update (my_sacas.__dict__)

    def kernel(self, mo_coeff=None, ci0=None, callback=None):
        return mcscf.mc1step_symm.CASSCF.kernel(self, mo_coeff, ci0, callback, newton_sacasscf.kernel)

    _finalize = SymAdaptedSACASSCF._finalize

