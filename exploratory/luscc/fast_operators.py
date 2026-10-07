"""Optional assembly optimizations for one-local-state excitation spaces.

The original MRH implementation is retained as a fallback for general spaces.
No Hamiltonian terms, thresholds, or summation order are changed.
"""
import numpy as np
from mrh.my_pyscf.lassi.op_o1.hams2ovlp import HamS2Ovlp


class CachedHamS2Ovlp(HamS2Ovlp):
    def __init__(self, *args, **kwargs):
        self._address_cache = {}
        self._sign_cache = {}
        super().__init__(*args, **kwargs)
        self._single_root = np.all(self.lroots == 1, axis=0)

    def _get_addr_range(self, raddr, *inv, _profile=True):
        key = (int(raddr), tuple(sorted(set(inv))))
        if key not in self._address_cache:
            if np.all(self.lroots[list(key[1]), raddr] == 1):
                value = self.offs_lroots[raddr, :1].copy()
            else:
                value = super()._get_addr_range(raddr, *inv, _profile=_profile)
            self._address_cache[key] = value
        return self._address_cache[key]

    def fermion_frag_shuffle(self, iroot, frags):
        key = (int(iroot), tuple(sorted(set(frags))))
        if key not in self._sign_cache:
            self._sign_cache[key] = super().fermion_frag_shuffle(iroot, frags)
        return self._sign_cache[key]

    def _get_spec_addr_ovlp_1space(self, rbra, rket, *inv):
        if not (self._single_root[rbra] and self._single_root[rket]):
            return super()._get_spec_addr_ovlp_1space(rbra, rket, *inv)
        inv = list(set(inv))
        fac = self.spin_shuffle[rbra] * self.spin_shuffle[rket]
        fac *= self.fermion_frag_shuffle(rbra, inv)
        fac *= self.fermion_frag_shuffle(rket, inv)
        value = np.asarray(fac, dtype=self.get_ci_dtype())[()]
        for f in range(self.nfrags):
            if f not in inv:
                value = self.ints[f].get_ovlp(rbra, rket)[0, 0] * value
        if abs(value) <= 1e-8:
            return np.empty(0, dtype=int), np.empty(0, dtype=int), np.empty(0, dtype=self.get_ci_dtype())
        if rbra == rket:
            value *= 0.5
        return (self.offs_lroots[rbra, :1], self.offs_lroots[rket, :1],
                np.asarray([value], dtype=self.get_ci_dtype()))

