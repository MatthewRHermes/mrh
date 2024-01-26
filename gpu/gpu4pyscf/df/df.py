#!/usr/bin/env python

return

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
