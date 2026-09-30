"""Exact total-spin constraints for LAS product-state expansions."""

from collections import defaultdict

import numpy as np
from scipy import linalg
from pyscf.fci import addons as fci_addons
from pyscf.fci.spin_op import contract_ss

from mrh.my_pyscf.lassi.citools import get_lroots, get_rootaddr_fragaddr


def _fermion_spin_shuffle(nelecs):
    """Phase from fragment-major to global spin-major operator ordering."""
    nperm = sum(sum(ne[0] for ne in nelecs[:frag]) * nelecs[frag][1]
                for frag in range(1, len(nelecs)))
    return -1.0 if nperm % 2 else 1.0


def _local_ladder(ci, norb, nelec, direction):
    """Apply a fragment S+ or S- operator to an FCI vector."""
    # The PySCF creation/annihilation helpers allocate real work arrays.
    if np.iscomplexobj(ci):
        real, out_nelec = _local_ladder(ci.real.copy(), norb, nelec, direction)
        imag, _ = _local_ladder(ci.imag.copy(), norb, nelec, direction)
        if real is None and imag is None:
            return None, out_nelec
        return (0 if real is None else real) + 1j * (0 if imag is None else imag), out_nelec
    na, nb = map(int, nelec)
    out_nelec = (na + 1, nb - 1) if direction == "+" else (na - 1, nb + 1)
    if min(out_nelec) < 0 or max(out_nelec) > norb:
        return None, out_nelec
    out = None
    for orb in range(norb):
        if direction == "+":
            term = fci_addons.des_b(ci, norb, (na, nb), orb)
            term = fci_addons.cre_a(term, norb, (na, nb - 1), orb)
        else:
            term = fci_addons.des_a(ci, norb, (na, nb), orb)
            term = fci_addons.cre_b(term, norb, (na - 1, nb), orb)
        out = term if out is None else out + term
    if out is None or not np.any(out):
        return None, out_nelec
    return out, out_nelec


def _product_basis(las):
    """Expand LASSI root spaces into explicit lists of fragment factors."""
    lroots = get_lroots(las.ci)
    rootaddr, fragaddr = get_rootaddr_fragaddr(lroots)
    nelec_frs = las.get_nelec_frs()
    basis = []
    for state, root in enumerate(rootaddr):
        factors, nelecs = [], []
        for frag in range(las.nfrags):
            block = las.ci[frag][root]
            factors.append(np.asarray(block[fragaddr[frag, state]]
                                      if block.ndim > 2 else block))
            nelecs.append(tuple(map(int, nelec_frs[frag, root])))
        basis.append((factors, tuple(nelecs)))
    return basis


def _residual_terms(factors, nelecs, norb_f, target_s2, actions=None):
    """Represent (S^2-target_s2)|product> as product-state terms."""
    terms = [(-target_s2, factors, nelecs)]
    source_shuffle = _fermion_spin_shuffle(nelecs)
    nfrag = len(factors)
    for a in range(nfrag):
        sfactors = list(factors)
        sfactors[a] = (contract_ss(factors[a], norb_f[a], nelecs[a])
                       if actions is None else
                       actions[a].apply(factors[a], nelecs[a], 'ss')[0])
        terms.append((1.0, sfactors, nelecs))
    for a in range(nfrag):
        ma = (nelecs[a][0] - nelecs[a][1]) / 2.0
        for b in range(a + 1, nfrag):
            mb = (nelecs[b][0] - nelecs[b][1]) / 2.0
            terms.append((2.0 * ma * mb, factors, nelecs))
            for da, db in (("+", "-"), ("-", "+")):
                if actions is None:
                    va, nea = _local_ladder(factors[a], norb_f[a], nelecs[a], da)
                    vb, neb = _local_ladder(factors[b], norb_f[b], nelecs[b], db)
                else:
                    va, nea = actions[a].apply(factors[a], nelecs[a], da)
                    vb, neb = actions[b].apply(factors[b], nelecs[b], db)
                if va is None or vb is None:
                    continue
                sfactors = list(factors)
                snelecs = list(nelecs)
                sfactors[a], sfactors[b] = va, vb
                snelecs[a], snelecs[b] = nea, neb
                snelecs = tuple(snelecs)
                # Local CI arrays use fragment-major spin ordering whereas
                # LASSI coefficients and the global FCI vector use spin-major
                # ordering. Spin ladders change this shuffle phase.
                phase = source_shuffle * _fermion_spin_shuffle(snelecs)
                terms.append((phase, sfactors, snelecs))
    return [(coef, fac, nel) for coef, fac, nel in terms if coef != 0]


def _product_overlap(left, right):
    lc, lf, ln = left
    rc, rf, rn = right
    if ln != rn:
        return 0.0
    value = np.conjugate(lc) * rc
    for bra, ket in zip(lf, rf):
        value *= np.vdot(bra, ket)
    return value


class _FragmentSpinFactors:
    """Call-local cache of exact fragment factors and their spin images.

    Equal copies are interned without numerical screening. No phases or small
    components are discarded, and a subsequent call observes mutated inputs.
    """

    def __init__(self, norb):
        self.norb = norb
        self.factors = defaultdict(list)
        self.indices = {}
        self.identities = {}
        self.images = {}

    def intern(self, ci, nelec):
        identity = (id(ci), nelec)
        if identity in self.identities:
            return self.identities[identity][1]
        key = (nelec, ci.dtype.str, ci.shape, ci.tobytes())
        if key not in self.indices:
            self.indices[key] = len(self.factors[nelec])
            self.factors[nelec].append(ci)
        index = self.indices[key]
        # Retain each object so Python cannot reuse its id during this call.
        self.identities[identity] = (ci, index)
        return index

    def apply(self, ci, nelec, operator):
        index = self.intern(ci, nelec)
        key = (nelec, index, operator)
        if key not in self.images:
            if operator == 'ss':
                # PySCF's contract_ss uses real work arrays.
                if np.iscomplexobj(ci):
                    image = (contract_ss(ci.real.copy(), self.norb, nelec)
                             + 1j * contract_ss(ci.imag.copy(), self.norb, nelec))
                else:
                    image = contract_ss(ci, self.norb, nelec)
                self.images[key] = image, nelec
            else:
                self.images[key] = _local_ladder(ci, self.norb, nelec, operator)
        return self.images[key]

    def overlaps(self):
        tables = {}
        for nelec, factors in self.factors.items():
            # Use the same scalar reduction as the original product overlap.
            # A GEMM changes last-bit rounding, which can rotate a degenerate
            # spin null space and affect the downstream Davidson initial guess.
            tables[nelec] = np.asarray([[np.vdot(a, b) for b in factors]
                                        for a in factors])
        return tables


def residual_gram(las, spin):
    """Return K_ij=<Q psi_i|Q psi_j> without global FCI vectors.

    Spin images and local overlap tables are computed once per distinct
    fragment factor. Product terms are grouped by electron sector and term
    position, then contracted in bounded tiles. Each group contains at most
    one term per input state, allowing indexed block accumulation without a
    Python loop over pairs of states or a dense matrix over all spin images.
    """
    if float(spin) < 0 or not np.isclose(2.0 * float(spin),
                                        round(2.0 * float(spin))):
        raise ValueError("target spin S must be a nonnegative integer or half-integer")
    target_s2 = float(spin) * (float(spin) + 1.0)
    basis = _product_basis(las)
    caches = [_FragmentSpinFactors(norb) for norb in las.ncas_sub]
    channels = defaultdict(lambda: defaultdict(list))
    for state, (factors, nelecs) in enumerate(basis):
        terms = _residual_terms(factors, nelecs, las.ncas_sub, target_s2,
                                actions=caches)
        for channel, (coef, factors, nelecs) in enumerate(terms):
            if any(not np.any(ci) for ci in factors):
                continue
            indices = [cache.intern(ci, nelec) for cache, ci, nelec
                       in zip(caches, factors, nelecs)]
            channels[channel][nelecs].append((state, coef, indices))
    tables = [cache.overlaps() for cache in caches]
    nstate = len(basis)
    dtype = np.result_type(np.float64, *[factor.dtype for state in basis
                             for factor in state[0]])
    gram = np.zeros((nstate, nstate), dtype=dtype)
    blocksize = 256
    groups = [{nelecs: (np.asarray([row[0] for row in rows]),
                        np.asarray([row[1] for row in rows]),
                        np.asarray([row[2] for row in rows]))
               for nelecs, rows in channels[channel].items()}
              for channel in sorted(channels)]
    # Preserve the original bra-term / ket-term summation order. In particular
    # do not combine diagonal terms or reorder sums by electron sector: even
    # roundoff-level changes can rotate the degenerate exact-spin null space.
    for agroup in groups:
        for bgroup in groups:
            for nelecs in agroup.keys() & bgroup.keys():
                astates, acoef, afactors = agroup[nelecs]
                bstates, bcoef, bfactors = bgroup[nelecs]
                for i in range(0, len(astates), blocksize):
                    ai = slice(i, i + blocksize)
                    for j in range(0, len(bstates), blocksize):
                        bj = slice(j, j + blocksize)
                        if astates[ai][-1] < bstates[bj][0]:
                            continue
                        value = acoef[ai, None].conj() * bcoef[None, bj]
                        value = value.astype(dtype, copy=False)
                        for f, nelec in enumerate(nelecs):
                            value *= tables[f][nelec][
                                afactors[ai, f, None], bfactors[None, bj, f]]
                        gram[np.ix_(astates[ai], bstates[bj])] += value
    gram = np.tril(gram)
    np.fill_diagonal(gram, gram.diagonal().real)
    return gram + np.tril(gram, -1).conj().T


def solve_exact_spin(hamiltonian, overlap, residual, spin, lin_dep_tol=1e-10,
                     spin_tol=1e-10):
    """Solve H in the metric-orthonormal numerical null space of K."""
    mval, mvec = linalg.eigh(overlap)
    mcut = max(lin_dep_tol, lin_dep_tol * max(1.0, float(mval[-1])))
    keep = mval > mcut
    if not np.any(keep):
        raise ValueError("LUSCC product basis is numerically linearly dependent")
    xmat = mvec[:, keep] / np.sqrt(mval[keep])
    korth = xmat.conj().T @ residual @ xmat
    kval, kvec = linalg.eigh((korth + korth.conj().T) / 2)
    # Eigenvalues of K are squared residual norms. ``spin_tol`` therefore
    # applies to their square roots.
    kscale = max(1.0, float(np.max(np.abs(kval))))
    kcut = max(spin_tol**2 * kscale,
               np.finfo(korth.real.dtype).eps * len(kval) * kscale * 10.0)
    null = np.abs(kval) <= kcut
    if not np.any(null):
        raise ValueError(
            f"No exact total-spin S={spin:g} vector exists in the LUSCC "
            f"space within tolerance {spin_tol:g}")
    projector = xmat @ kvec[:, null]
    hcon = projector.conj().T @ hamiltonian @ projector
    energy, vec = linalg.eigh((hcon + hcon.conj().T) / 2)
    coeff = projector @ vec
    residual_norm = np.sqrt(np.maximum(0.0, np.real(np.einsum(
        "ip,ij,jp->p", coeff.conj(), residual, coeff))))
    return energy, coeff, residual_norm, kval


def exact_spin_basis(overlap, residual, spin, lin_dep_tol=1e-10,
                     spin_tol=1e-10):
    """Return a raw-coefficient orthonormal basis for the exact-spin space."""
    mval, mvec = linalg.eigh((overlap + overlap.conj().T) / 2)
    mcut = max(lin_dep_tol, lin_dep_tol * max(1.0, float(mval[-1])))
    keep = mval > mcut
    if not np.any(keep):
        raise ValueError("LUSCC product basis is numerically linearly dependent")
    xmat = mvec[:, keep] / np.sqrt(mval[keep])
    korth = xmat.conj().T @ residual @ xmat
    kval, kvec = linalg.eigh((korth + korth.conj().T) / 2)
    kscale = max(1.0, float(np.max(np.abs(kval))))
    kcut = max(spin_tol**2 * kscale,
               np.finfo(korth.real.dtype).eps * len(kval) * kscale * 10.0)
    null = np.abs(kval) <= kcut
    if not np.any(null):
        raise ValueError(
            f"No exact total-spin S={spin:g} vector exists in the LUSCC "
            f"space within tolerance {spin_tol:g}")
    return xmat @ kvec[:, null], kval
