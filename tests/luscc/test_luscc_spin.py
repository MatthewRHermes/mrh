import numpy as np
import pytest
from unittest import mock
from pyscf.fci import cistring
from pyscf.fci.spin_op import contract_ss

from mrh.exploratory.luscc import spin as spin_module
from mrh.exploratory.luscc.spin import (
    residual_gram, _product_basis, _product_overlap, _residual_terms)
from mrh.my_pyscf.lassi import op_o0


class _SpinImpureProducts:
    nfrags = 2
    ncas_sub = (2, 2)

    def __init__(self):
        # Fixed-Ms fragment vectors deliberately mixing local singlet and
        # triplet components. No fragment spin quantum number is supplied.
        a = np.array([[0.0, 1.0], [2.0, 0.0]]) / np.sqrt(5.0)
        b = np.array([[0.0, 3.0], [-1.0, 0.0]]) / np.sqrt(10.0)
        c = np.array([[1.0, 0.0], [0.0, 2.0]]) / np.sqrt(5.0)
        d = np.array([[2.0, 0.0], [0.0, -1.0]]) / np.sqrt(5.0)
        self.ci = [[a, c], [b, d]]
        self._nelec = np.ones((2, 2, 2), dtype=int)

    def get_nelec_frs(self):
        return self._nelec


def test_residual_gram_matches_combined_fci_for_spin_impure_fragments():
    products = _SpinImpureProducts()
    k_factorized = residual_gram(products, spin=0)

    global_ci, nelec = op_o0.ci_outer_product(
        products.ci, products.ncas_sub, products.get_nelec_frs())
    qci = [contract_ss(ci, sum(products.ncas_sub), tuple(nel))
           for ci, nel in zip(global_ci, nelec)]
    k_fci = np.asarray([[np.vdot(left, right) for right in qci]
                        for left in qci])
    np.testing.assert_allclose(k_factorized, k_fci, atol=1e-12)


class _SectorProducts:
    """Small complete-CAS oracle with charge/spin hops and local-root blocks."""

    def __init__(self, nfrags, complex_ci=False):
        self.nfrags = nfrags
        self.ncas_sub = (2,) * nfrags
        neutral = [(1, 1)] * nfrags
        sectors = [neutral, [(2, 0), (0, 2)] + neutral[2:],
                   [(2, 1), (0, 1)] + neutral[2:], neutral]
        self._nelec = np.asarray(sectors).transpose(1, 0, 2)
        rng = np.random.default_rng(102)
        self.ci = []
        for f in range(nfrags):
            roots = []
            for r, sector in enumerate(sectors):
                shape = tuple(cistring.num_strings(2, ne) for ne in sector[f])
                # A Cartesian product of two local-root blocks in rootspace 0.
                nlocal = 2 if r == 0 and f < 2 else 1
                ci = rng.normal(size=(nlocal,) + shape)
                if complex_ci:
                    ci = ci + 1j * rng.normal(size=ci.shape)
                ci /= np.linalg.norm(ci.reshape(nlocal, -1), axis=1)[:, None, None]
                roots.append(ci if nlocal > 1 else ci[0])
            self.ci.append(roots)

    def get_nelec_frs(self):
        return self._nelec


@pytest.mark.parametrize('nfrags,spin,complex_ci',
                         [(2, 0, False), (3, 1, False), (4, 2, False),
                          (2, 1, True), (4, 0, True)])
def test_cached_residual_matches_global_fci(nfrags, spin, complex_ci):
    products = _SectorProducts(nfrags, complex_ci)
    original = [[ci.copy() for ci in roots] for roots in products.ci]
    global_ci, nelecs = op_o0.ci_outer_product(
        products.ci, products.ncas_sub, products.get_nelec_frs())
    residuals = []
    for ci, ne in zip(global_ci, nelecs):
        qci = contract_ss(ci.real.copy(), 2*nfrags, tuple(ne))
        if complex_ci:
            qci = qci + 1j * contract_ss(ci.imag.copy(), 2*nfrags, tuple(ne))
        residuals.append((qci - spin*(spin+1)*ci).ravel())
    residuals = np.stack(residuals)
    expected = residuals.conj() @ residuals.T
    actual = residual_gram(products, spin)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(actual, actual.conj().T, atol=1e-14, rtol=0)
    assert np.linalg.eigvalsh(actual).min() > -1e-12
    for new, old in zip(products.ci, original):
        for ci, ci0 in zip(new, old):
            np.testing.assert_array_equal(ci, ci0)


def test_spin_images_reused_for_equal_copies_and_cache_is_call_local():
    products = _SpinImpureProducts()
    products.ci = [[roots[0].copy() for _ in range(5)] for roots in products.ci]
    products._nelec = np.ones((2, 5, 2), dtype=int)
    with mock.patch.object(spin_module, 'contract_ss',
                           wraps=spin_module.contract_ss) as ss:
        result = residual_gram(products, 0)
    assert ss.call_count == 2  # One distinct local factor on each fragment.
    np.testing.assert_allclose(result, result[0, 0], atol=1e-12)
    products.ci[0][0][0, 0] += 0.5
    updated = residual_gram(products, 0)
    assert not np.allclose(result, updated)


def test_residual_tiling_preserves_repeated_states():
    products = _SpinImpureProducts()
    small = residual_gram(products, 0)
    # Cross the 256-row tile boundary, including noncontiguous source indices.
    products.ci = [[roots[i % 2] for i in range(259)] for roots in products.ci]
    products._nelec = np.ones((2, 259, 2), dtype=int)
    result = residual_gram(products, 0)
    indices = np.arange(259) % 2
    np.testing.assert_allclose(result, small[np.ix_(indices, indices)],
                               atol=1e-12, rtol=0)


def test_half_integer_spin_with_charge_transfer():
    products = _SpinImpureProducts()
    products.ci = [[np.asarray([[1.], [0.]]), products.ci[0][0]],
                   [products.ci[1][0], np.asarray([[0.], [1.]])]]
    products._nelec = np.asarray([[(1, 0), (1, 1)], [(1, 1), (1, 0)]])
    global_ci, nelecs = op_o0.ci_outer_product(
        products.ci, products.ncas_sub, products.get_nelec_frs())
    q = np.stack([(contract_ss(ci, 4, tuple(ne)) - 0.75*ci).ravel()
                  for ci, ne in zip(global_ci, nelecs)])
    np.testing.assert_allclose(residual_gram(products, 0.5), q.conj() @ q.T,
                               atol=1e-12, rtol=0)


@pytest.mark.parametrize('nfrags,spin', [(2, 0), (3, 1), (4, 9)])
def test_real_residual_preserves_scalar_summation_bits(nfrags, spin):
    products = _SectorProducts(nfrags)
    terms = [_residual_terms(*state, products.ncas_sub, spin*(spin+1))
             for state in _product_basis(products)]
    expected = np.zeros((len(terms), len(terms)))
    for i in range(len(terms)):
        for j in range(i+1):
            expected[i, j] = sum(_product_overlap(a, b)
                                 for a in terms[i] for b in terms[j])
            expected[j, i] = expected[i, j]
    np.testing.assert_array_equal(residual_gram(products, spin), expected)


@pytest.mark.parametrize('spin', [-1, 0.2])
def test_invalid_target_spin(spin):
    with pytest.raises(ValueError, match='nonnegative integer or half-integer'):
        residual_gram(_SpinImpureProducts(), spin)
