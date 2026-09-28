#!/usr/bin/env python
from mrh.my_pyscf.pbc.mcscf.mc_ao2mo import _ERIS as KCASSCF_ERIS

# Author: Bhavnesh Jangid

class _ERIS(KCASSCF_ERIS):
    '''Periodic AO2MO intermediates required by k-LASSCF.

    The k-CASSCF parent already constructs the ``ppaa`` integrals.  The
    additional ``paaa`` block used by the LASSCF orbital gradient is exposed
    as an active-space slice, avoiding another integral transformation or
    disk-backed tensor.
    '''

    def __init__(self, klasscf, mo_kpts, method='disk', level=2):
        super().__init__(klasscf, mo_kpts, method=method, level=level)
        self.ncore = klasscf.ncore
        self.nocc = klasscf.ncore + klasscf.ncas

    def get_paaa(self, k1, k2, k3):
        ppaa = self.get_ppaa(k1, k2, k3)
        return ppaa[:, self.ncore:self.nocc, :, :]

    paaa = get_paaa
