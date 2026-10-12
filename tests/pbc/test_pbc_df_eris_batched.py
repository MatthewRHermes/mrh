"""Regression tests for cached, signed, batched DF ERIs and core terms."""

import sys
import unittest
from unittest.mock import patch

import numpy as np
from pyscf.pbc import gto
from pyscf.pbc.lib import kpts_helper

from mrh.my_pyscf.pbc.mcscf.mc_ao2mo_opt import build_cas_eris, build_eris, _PairCache, _channel, _transfer_groups
from mrh.my_pyscf.pbc.mcscf.mc_ao2mo import _ERIS


class _SignedDF:
    max_memory, blockdim, _cderi = 100, 3, True

    def __init__(self, cell, kpts, rng):
        self.cell, self.kpts, self.opens = cell, kpts, 0
        self.factors = {}
        self.signs = np.array([1, 1, 1, 1, -1, -1], dtype=np.int8)
        for ki in range(len(kpts)):
            for kj in range(ki, len(kpts)):
                value = rng.standard_normal((6, 3, 3)) + 1j * rng.standard_normal((6, 3, 3))
                if ki == kj:
                    value += value.transpose(0, 2, 1).conj()
                self.factors[ki, kj] = value
                self.factors[kj, ki] = value.transpose(0, 2, 1).conj()

    def sr_loop(self, kpts, max_memory, compact, blksize):
        self.opens += 1
        ki, kj = [int(np.argmin(np.linalg.norm(self.kpts - point, axis=1))) for point in kpts]
        array = self.factors[ki, kj]
        # The positive and negative parts have separate auxiliary blocks.
        for begin, end, sign in ((0, 4, 1), (4, 6, -1)):
            for start in range(begin, end, blksize):
                block = array[start:min(end, start + blksize)].reshape(-1, 9)
                yield block.real.copy(), block.imag.copy(), sign


class KnownValues(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(123)
        cell = gto.Cell(atom='He 0 0 0', basis='sto-3g', a=np.eye(3)*4, verbose=0).build()
        self.kpts = cell.make_kpts([2, 1, 1])
        self.df = _SignedDF(cell, self.kpts, self.rng)
        self.mo = np.array([np.linalg.qr(self.rng.standard_normal((3, 3)) + 1j*self.rng.standard_normal((3, 3)))[0]
                            for _ in self.kpts])
        self.z = {pair: np.einsum('up,Luv,vq->Lpq', self.mo[pair[0]].conj(), value, self.mo[pair[1]])
                  for pair, value in self.df.factors.items()}
        scf = type('SCF', (), {'with_df': self.df, 'kpts': self.kpts, 'cell': cell})()
        self.kcas = type('KCAS', (), {'_scf': scf, 'max_memory': 10000, 'verbose': 0, 'stdout': sys.stdout,
                                    'ncore': 1, 'ncas': 1, 'get_hcore': lambda obj: np.zeros((2, 3, 3))})()
        self.kconserv = kpts_helper.get_kconserv(cell, self.kpts)

    def reference_blocks(self, output):
        slices = {'ppaa': (slice(None), slice(None), slice(1, 2), slice(1, 2)),
                  'papa': (slice(None), slice(1, 2), slice(None), slice(1, 2)),
                  'paap': (slice(None), slice(1, 2), slice(1, 2), slice(None))}
        for name, indices in slices.items():
            for k1, k2, k3 in np.ndindex((2,) * 3):
                k4 = self.kconserv[k1, k2, k3]
                left = self.z[k1, k2][:, indices[0], indices[1]]
                right = self.z[k3, k4][:, indices[2], indices[3]]
                expected = np.einsum('Lpq,Lrs,L->pqrs', left, right, self.df.signs) / 2
                np.testing.assert_allclose(output[name][k1, k2, k3], expected, atol=1e-12, rtol=1e-12)

    def test_active_eris_signed_ram_and_spill(self):
        for memory in (10000, 0):
            with self.subTest(max_memory=memory):
                self.kcas.max_memory = memory
                self.df.opens = 0
                output = build_cas_eris(self.kcas, self.mo, self.kconserv)
                self.assertEqual(self.df.opens, 4)
                for k1, k2, k3 in np.ndindex((2,) * 3):
                    k4 = self.kconserv[k1, k2, k3]
                    expected = np.einsum(
                        'Lpq,Lrs,L->pqrs', self.z[k1, k2],
                        self.z[k3, k4], self.df.signs,
                    ) / 2
                    np.testing.assert_allclose(
                        output[k1, k2, k3], expected, atol=1e-12, rtol=1e-12,
                    )

    def test_signed_core_and_all_channels(self):
        for disk in (False, True):
            with self.subTest(disk=disk):
                self.df.opens = 0
                output, j_pc, k_pc, hcore = build_eris(self.kcas, self.mo, 1, 1, disk=disk)
                try:
                    self.assertEqual(self.df.opens, 4)
                    self.reference_blocks(output)
                    for k in range(2):
                        z = self.z[k, k]
                        diagonal = np.diagonal(z, axis1=1, axis2=2)
                        j = np.einsum('Lp,Lc,L->pc', diagonal, diagonal[:, :1], self.df.signs) / 2
                        exchange = np.einsum('Lpc,Lpc,L->pc', z[:, :, :1], z[:, :, :1], self.df.signs) / 6
                        exchange += np.einsum('Lpc,Lcp,L->pc', z[:, :, :1], z[:, :1, :], self.df.signs).conj() / 3
                        np.testing.assert_allclose(j_pc[k], j, atol=1e-12)
                        np.testing.assert_allclose(k_pc[k], exchange, atol=1e-12)
                    if disk:
                        operator = _ERIS.__new__(_ERIS)
                        operator.erifile = output
                        operator.ppaa_kpts = operator.papa_kpts = operator.paap_kpts = None
                        np.testing.assert_array_equal(operator.get_ppaa(0, 1, 0), output['ppaa'][0, 1, 0])
                finally:
                    if disk:
                        output.close()

    def test_spill_and_small_auxiliary_batches(self):
        cache = _PairCache(1, 3)
        try:
            for pair, value in self.z.items():
                cache.append(pair, value[:4], 1)
                cache.append(pair, value[4:], -1)
            cache.finish()
            self.assertIsNotNone(cache.file)
            groups = _transfer_groups(self.kconserv, cache)
            output = np.zeros((2, 2, 2, 3, 3, 1, 1), dtype=complex)
            _channel(cache, groups, output, (3, 3, 1, 1), (slice(None), slice(None)),
                     (slice(1, 2), slice(1, 2)), 2, workspace_bytes=128)
            for k1, k2, k3 in np.ndindex((2,) * 3):
                k4 = self.kconserv[k1, k2, k3]
                expected = np.einsum('Lpq,Lrs,L->pqrs', self.z[k1, k2], self.z[k3, k4][:, 1:2, 1:2], self.df.signs) / 2
                np.testing.assert_allclose(output[k1, k2, k3], expected, atol=1e-12)
        finally:
            cache.close()

    def test_level_two_skips_core_terms(self):
        output, j_pc, k_pc, _ = build_eris(self.kcas, self.mo, 1, 1, level=2)
        self.assertIsNone(j_pc)
        self.assertIsNone(k_pc)
        self.reference_blocks(output)


if __name__ == '__main__':
    unittest.main()
