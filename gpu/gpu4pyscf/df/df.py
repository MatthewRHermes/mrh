#!/usr/bin/env python

#return

import copy
import tempfile
import contextlib
import numpy
import h5py
from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.df import df
from pyscf.df import incore
from pyscf.df import outcore
from pyscf.df import r_incore
from pyscf.df import addons
from pyscf.df import df_jk
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo.incore import _conc_mos, iden_coeffs
from pyscf.ao2mo.outcore import _load_from_h5g
from pyscf import __config__

from gpu4pyscf.lib.utils import patch_cpu_kernel

class DF(df.DF):
    
    def build(self):
        print("Inside gpu4pyscf/df/df.py::build()  self= ", hex(id(self)))
        t0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)

        self.check_sanity()
        self.dump_flags()

        mol = self.mol
        auxmol = self.auxmol = addons.make_auxmol(self.mol, self.auxbasis)
        nao = mol.nao_nr()
        naux = auxmol.nao_nr()
        nao_pair = nao*(nao+1)//2
    
        print("Inside df.py::build() w/ self= ", id(self))
        is_custom_storage = isinstance(self._cderi_to_save, str)
        max_memory = self.max_memory - lib.current_memory()[0]
        int3c = mol._add_suffix('int3c2e')
        int2c = mol._add_suffix('int2c2e')
        if (nao_pair*naux*8/1e6 < .9*max_memory and not is_custom_storage):
            self._cderi = incore.cholesky_eri(mol, int3c=int3c, int2c=int2c,
                                              auxmol=auxmol,
                                              max_memory=max_memory, verbose=log)
        else:
            if is_custom_storage:
                cderi = self._cderi_to_save
            else:
                cderi = self._cderi_to_save.name
                
            if isinstance(self._cderi, str):
                # If cderi needs to be saved in
                log.warn('Value of _cderi is ignored. DF integrals will be '
                         'saved in file %s .', cderi)
                
            if self._compatible_format:
                outcore.cholesky_eri(mol, cderi, dataname=self._dataname,
                                     int3c=int3c, int2c=int2c, auxmol=auxmol,
                                     max_memory=max_memory, verbose=log)
            else:
                # Store DF tensor in blocks. This is to reduce the
                # initiailzation overhead
                outcore.cholesky_eri_b(mol, cderi, dataname=self._dataname,
                                       int3c=int3c, int2c=int2c, auxmol=auxmol,
                                       max_memory=max_memory, verbose=log)
            self._cderi = cderi
            log.timer_debug1('Generate density fitting integrals', *t0)
        return self
    def ao2mo(self, mo_coeffs,
              compact=getattr(__config__, 'df_df_DF_ao2mo_compact', True)):
        print("Inside gpu4pyscf/df/df.py::ao2mo()")#  self= ", hex(id(self)))
        if isinstance(mo_coeffs, numpy.ndarray) and mo_coeffs.ndim == 2:
            mo_coeffs = (mo_coeffs,) * 4
        ijmosym, nij_pair, moij, ijslice = _conc_mos(mo_coeffs[0], mo_coeffs[1], compact)
        print(ijslice)
        klmosym, nkl_pair, mokl, klslice = _conc_mos(mo_coeffs[2], mo_coeffs[3], compact)
        mo_eri = numpy.zeros((nij_pair,nkl_pair))
        sym = (iden_coeffs(mo_coeffs[0], mo_coeffs[2]) and
               iden_coeffs(mo_coeffs[1], mo_coeffs[3]))
        Lij = Lkl = None
        for eri1 in self.loop():
            Lij = _ao2mo.nr_e2(eri1, moij, ijslice, aosym='s2', mosym=ijmosym, out=Lij)
            if sym:
                #print("Here")
                Lkl = Lij
            else:
                Lkl = _ao2mo.nr_e2(eri1, mokl, klslice, aosym='s2', mosym=klmosym, out=Lkl)
            lib.dot(Lij.T, Lkl, 1, mo_eri, 1)
        return mo_eri
    get_mo_eri = ao2mo
