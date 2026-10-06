"""Spin-separated LASSI 3-RDMs from fragment annihilation images.

No full-CAS product vectors are formed. Local annihilation images and overlaps
are reused, and products are contracted in bounded tiles. The returned dense
3-RDM still requires O(norb**6) storage.
"""
from collections import defaultdict
from itertools import combinations, product

import numpy as np
from pyscf.fci import addons
from mrh.my_pyscf.lassi.citools import get_lroots, get_rootaddr_fragaddr



class _Factors:
    def __init__(self, norb):
        self.norb = int(norb)
        self.factors = defaultdict(list)
        self.indices = {}
        self.images = {}

    def intern(self, ci, nelec):
        ci = np.ascontiguousarray(ci)
        key = (nelec, ci.shape, ci.dtype.str, ci.tobytes())
        if key not in self.indices:
            self.indices[key] = len(self.factors[nelec])
            self.factors[nelec].append(ci)
        return self.indices[key]

    def image(self, ci, nelec, ops):
        source = self.intern(ci, nelec)
        key = (nelec, source, ops)
        if key not in self.images:
            out = ci
            target = list(nelec)
            for orbital, spin in ops:
                if target[spin] == 0:
                    self.images[key] = None
                    return None
                destroy = addons.des_a if spin == 0 else addons.des_b
                if np.iscomplexobj(out):
                    out = (destroy(out.real.copy(), self.norb, tuple(target), orbital)
                           + 1j * destroy(out.imag.copy(), self.norb, tuple(target), orbital))
                else:
                    out = destroy(out, self.norb, tuple(target), orbital)
                target[spin] -= 1
            if np.any(out):
                target = tuple(target)
                self.images[key] = (target, self.intern(out, target))
            else:
                self.images[key] = None
        return self.images[key]

    def overlaps(self):
        return {sector: (vectors.conj() @ vectors.T)
                for sector, factors in self.factors.items()
                for vectors in [np.stack([ci.ravel() for ci in factors])]}


def _triples(norb, nalpha):
    return [a + b for a in combinations(range(norb), nalpha)
            for b in combinations(range(norb, 2*norb), 3-nalpha)]


def _ordered_map(norb, nalpha, triples):
    """Antisymmetric expansion from unique triples to ordered spatial indices."""
    lookup = {triple: k for k, triple in enumerate(triples)}
    indices = np.zeros(norb**3, dtype=int)
    signs = np.zeros(norb**3, dtype=int)
    for k, spatial in enumerate(product(range(norb), repeat=3)):
        spin_orbs = tuple(p + (0 if j < nalpha else norb)
                          for j, p in enumerate(spatial))
        if len(set(spin_orbs)) < 3:
            continue
        indices[k] = lookup[tuple(sorted(spin_orbs))]
        inversions = sum(spin_orbs[i] > spin_orbs[j]
                         for i in range(3) for j in range(i+1, 3))
        signs[k] = (-1)**inversions
    return indices, signs


def roots_make_rdm3s(las, ci_fr, nelec_frs, si, blocksize=128, **kwargs):
    """Return (nstate, 4, norb, ..., norb) blocks aaa, aab, abb, bbb.

    dm3[p,q,r,s,t,u] = <p† r† t† u s q>, with the indicated
    spins on (p,q), (r,s), (t,u). All SI products and local-root blocks
    participate; different residual fragment electron sectors are orthogonal.
    """
    if blocksize < 1:
        raise ValueError('blocksize must be positive')
    norb_f = np.asarray(las.ncas_sub, dtype=int)
    norb = int(sum(norb_f))
    starts = np.r_[0, np.cumsum(norb_f)]
    rootaddr, fragaddr = get_rootaddr_fragaddr(get_lroots(ci_fr))
    si = np.asarray(si)
    if si.shape[0] != len(rootaddr):
        raise ValueError('SI coefficients do not match the product basis')
    basis = []
    for state, root in enumerate(rootaddr):
        factors = [block[fragaddr[f, state]] if block.ndim > 2 else block
                   for f in range(len(norb_f)) for block in [ci_fr[f][root]]]
        nelecs = tuple(tuple(map(int, nelec_frs[f, root])) for f in range(len(norb_f)))
        basis.append((factors, nelecs))
    dtype = np.result_type(si.dtype, *[v.dtype for fac, _ in basis for v in fac])
    result = np.zeros((si.shape[1], 4) + (norb,)*6, dtype=dtype)
    caches = [_Factors(n) for n in norb_f]
    for channel, nalpha in enumerate((3, 2, 1, 0)):
        triples = _triples(norb, nalpha)
        if not triples:
            continue
        groups = defaultdict(list)
        for t, triple in enumerate(triples):
            local_ops = [[] for _ in norb_f]
            global_ops = []
            for orbital in triple:
                spatial, spin = orbital % norb, orbital // norb
                f = int(np.searchsorted(starts, spatial, side='right')-1)
                local_ops[f].append((spatial-int(starts[f]), spin))
                global_ops.append((f, spin))
            for state, (factors, nelecs) in enumerate(basis):
                if not np.any(si[state]):
                    continue
                remaining = [list(n) for n in nelecs]
                phase = 1
                # PySCF destruction signs count same-spin occupied orbitals
                # above the orbital; des_b also includes local alpha parity.
                for f, spin in global_ops:
                    prefix = sum(n[spin] for n in remaining[f+1:])
                    if spin == 1:
                        prefix += sum(n[0] for g, n in enumerate(remaining) if g != f)
                    phase *= (-1)**prefix
                    remaining[f][spin] -= 1
                target, ids = [], []
                for f, cache in enumerate(caches):
                    image = cache.image(factors[f], nelecs[f], tuple(local_ops[f]))
                    if image is None:
                        break
                    sector, index = image
                    target.append(sector)
                    ids.append(index)
                else:
                    target = tuple(target)
                    groups[target].append((t, state, phase, ids))
        tables = [cache.overlaps() for cache in caches]
        gamma = np.zeros((si.shape[1], len(triples), len(triples)), dtype=dtype)
        for sector, rows in groups.items():
            tid = np.array([r[0] for r in rows])
            coefficients = si[[r[1] for r in rows]] * np.array([r[2] for r in rows])[:, None]
            ids = np.array([r[3] for r in rows])
            for i in range(0, len(rows), blocksize):
                ai = slice(i, i+blocksize)
                for j in range(0, len(rows), blocksize):
                    bj = slice(j, j+blocksize)
                    overlap = np.ones((len(tid[ai]), len(tid[bj])), dtype=dtype)
                    for f, nelec in enumerate(sector):
                        overlap *= tables[f][nelec][ids[ai, f, None], ids[None, bj, f]]
                    addresses = tid[ai, None]*len(triples) + tid[None, bj]
                    for root in range(si.shape[1]):
                        values = overlap * coefficients[ai, root, None].conj() * coefficients[None, bj, root]
                        np.add.at(gamma[root].ravel(), addresses.ravel(), values.ravel())
        indices, signs = _ordered_map(norb, nalpha, triples)
        for root in range(si.shape[1]):
            ordered = gamma[root][indices[:, None], indices[None, :]]
            ordered *= signs[:, None] * signs[None, :]
            result[root, channel] = ordered.reshape((norb,)*6).transpose(0, 3, 1, 4, 2, 5)
    return result


def root_make_rdm3s(las, ci_fr, nelec_frs, si, ix, **kwargs):
    return roots_make_rdm3s(las, ci_fr, nelec_frs, si[:, ix:ix+1], **kwargs)[0]
