"""Independent full-CAS oracles for fragment-factorized three-body densities."""
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
from pyscf.fci import cistring, addons, rdm
from mrh.my_pyscf.lassi import op_o0
from mrh.my_pyscf.lassi.op_o1 import rdm3


@pytest.mark.parametrize('nfrags', [2, 3])
def test_fragment_rdm3_matches_full_cas(nfrags):
    rng = np.random.default_rng(42)
    las = SimpleNamespace(ncas_sub=(2,)*nfrags)
    sectors = [[(1,1)]*nfrags, [(2,0),(0,2)]+[(1,1)]*(nfrags-2),
               [(2,1),(0,1)]+[(1,1)]*(nfrags-2)]
    nelec = np.array(sectors).transpose(1,0,2)
    ci = []
    for f in range(nfrags):
        local = []
        for r, sector in enumerate(sectors):
            shape = tuple(cistring.num_strings(2, n) for n in sector[f])
            vectors = rng.normal(size=(2 if r == 0 and f == 0 else 1,)+shape)
            vectors /= np.linalg.norm(vectors.reshape(len(vectors),-1),axis=1)[:,None,None]
            local.append(vectors if len(vectors)>1 else vectors[0])
        ci.append(local)
    si = rng.normal(size=(4,2))
    si /= np.linalg.norm(si,axis=0)
    expected = op_o0.roots_make_rdm3s(las, ci, nelec, si)
    # Production path must never construct global product vectors or call the FCI RDM kernel.
    with mock.patch.object(op_o0, 'ci_outer_product', side_effect=AssertionError('global CI')), \
         mock.patch.object(rdm, 'make_dm123', side_effect=AssertionError('FCI RDM kernel')):
        actual = rdm3.roots_make_rdm3s(las, ci, nelec, si, blocksize=3)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


def test_complex_fragment_rdm3_against_global_annihilation():
    rng = np.random.default_rng(81)
    las = SimpleNamespace(ncas_sub=(2,2))
    ci = [[rng.normal(size=shape)+1j*rng.normal(size=shape)
           for shape in [(2,2), (1,2)]] for _ in range(2)]
    # ci_outer_product is real-only in some PySCF versions: build this simple
    # neutral product explicitly in spin-major determinant order.
    from itertools import product
    strings = [1,2]
    addresses = {int(s): i for i,s in enumerate(cistring.make_strings(range(4),2))}
    psi = np.zeros((6,6),dtype=complex)
    for a,b,c,d in product(range(2),repeat=4):
        psi[addresses[strings[a]|(strings[c]<<2)], addresses[strings[b]|(strings[d]<<2)]] = ci[0][0][a,b]*ci[1][0][c,d]
    coeff = np.array([[.7+.2j], [.3-.4j]])
    psi *= coeff[0,0]
    for b,d in product(range(2),repeat=2):
        psi[addresses[3], addresses[strings[b]|(strings[d]<<2)]] += (
            coeff[1,0]*ci[0][1][0,b]*ci[1][1][0,d])
    nelec = np.array([[[1,1], [2,1]],[[1,1], [0,1]]])
    actual = rdm3.roots_make_rdm3s(las,ci,nelec,coeff)[0]
    for channel, spins in enumerate([(0,0,0),(0,0,1),(0,1,1),(1,1,1)]):
        images = []
        for orbitals in product(range(4),repeat=3):
            v = psi
            target = [2,2]
            for p,s in zip(orbitals,spins):
                des = addons.des_a if s == 0 else addons.des_b
                if target[s] == 0:
                    v = None
                    break
                v = des(v.real.copy(),4,tuple(target),p)+1j*des(v.imag.copy(),4,tuple(target),p)
                target[s] -= 1
            images.append(np.zeros(1) if v is None else v.ravel())
        if any(v.size != images[0].size for v in images):
            expected = np.zeros((4,)*6,dtype=complex)
        else:
            vectors = np.stack(images)
            expected = (vectors.conj() @ vectors.T).reshape((4,)*6).transpose(0,3,1,4,2,5)
        np.testing.assert_allclose(actual[channel],expected,atol=1e-11,rtol=0)


def test_lassi_public_rdm3_uses_fragments(h4_lassis):
    expected = h4_lassis.make_casdm3s(state=0, opt=0)
    with mock.patch.object(op_o0, 'ci_outer_product', side_effect=AssertionError('global CI')), \
         mock.patch.object(rdm, 'make_dm123', side_effect=AssertionError('FCI RDM kernel')):
        actual = h4_lassis.make_casdm3s(state=0)
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)


def test_public_rdm3_weights_and_invalid_backend(h4_lassis):
    coefficients = h4_lassis.si[:, :2]
    expected = h4_lassis.make_casdm3s(si=coefficients, opt=0)
    actual = h4_lassis.make_casdm3s(si=coefficients, weights=[.3, .7])
    np.testing.assert_allclose(actual, np.tensordot([.3, .7], expected, axes=1),
                               atol=1e-12, rtol=0)
    with pytest.raises(ValueError, match='opt must be'):
        h4_lassis.make_casdm3s(state=0, opt=2)
