"""Cached DF pair transforms and batched periodic orbital-response ERIs.

The RAM and HDF5 output paths share exactly the same contractions. Disk
output uses three datasets instead of one dataset per k-point triple.
Pair factors spill to a temporary HDF5 file when their RAM budget is full.
"""

# Author: Bhavnesh Jangid

import numpy as np

from pyscf import lib
from pyscf.ao2mo import _ao2mo
from pyscf.pbc.df.df_ao2mo import _conc_mos, gamma_point
from pyscf.pbc.lib import kpts_helper
from mrh.my_pyscf.pbc.util.checks import _check_shape

class _PairCache:
    '''Store MO-transformed DF three-index factors L[Q,p,q] per k-point pair.

    Each key is a pair (ki, kj); its value is the auxiliary-index-ordered
    list of complex (naux_chunk, nmo, nmo) blocks produced by
    ``_transform_pairs``. Blocks are held in RAM until their total size
    would exceed ``max_mem``; then everything is moved to a temporary
    HDF5 file (one growable (naux, nmo, nmo) dataset per pair) and later
    blocks are written there directly. ``max_mem == 0`` starts on disk.

    Attributes:
        max_mem : RAM budget for cached blocks, in MB.
        nmo : orbital dimension of both MO indices.
        arrays : {pair: [ndarray, ...]} in-RAM blocks (empty once spilled).
        file : None while in RAM, else the H5TmpFile.
        mem : MB currently held in ``arrays``.
        naux : {pair: total auxiliary rows stored}.
        signs : {pair: int8 array (naux,)} sign of each auxiliary row from
            ``sr_loop`` (+1/-1); available after :meth:`finish`.

    Usage: :meth:`append` blocks, call :meth:`finish`, then :meth:`read`
    row ranges/orbital slices, and :meth:`close` when done.
    '''
    def __init__(self, max_mem, nmo):
        self.max_mem = max_mem
        self.nmo = nmo
        self.arrays = {}
        self.file = None
        self.mem = 0
        self.naux = {}
        self.signs = {}
        self._sign_chunks = {}
        if max_mem == 0:
            self._spill()

    @staticmethod
    def _key(pair):
        return f'{pair[0]}_{pair[1]}'

    def _dataset(self, pair):
        key = self._key(pair)
        if key not in self.file:
            self.file.create_dataset(key, shape=(0, self.nmo, self.nmo),
                                     maxshape=(None, self.nmo, self.nmo), dtype=complex)
        return self.file[key]

    def _write(self, pair, array):
        dataset = self._dataset(pair)
        start = dataset.shape[0]
        dataset.resize(start + len(array), axis=0)
        dataset[start:] = array

    def _spill(self):
        self.file = lib.H5TmpFile()
        for pair, chunks in self.arrays.items():
            for array in chunks:
                self._write(pair, array)
        self.arrays.clear()
        self.mem = 0

    def append(self, pair, array, sign):
        self.naux[pair] = self.naux.get(pair, 0) + len(array)
        self._sign_chunks.setdefault(pair, []).append(np.full(len(array), sign, dtype=np.int8))
        if self.file is None and self.mem + array.nbytes / 1e6 > self.max_mem:
            self._spill()
        if self.file is None:
            self.arrays.setdefault(pair, []).append(array)
            self.mem += array.nbytes / 1e6
        else:
            self._write(pair, array)

    def finish(self):
        self.signs = {pair: np.concatenate(chunks) 
                      for pair, chunks in self._sign_chunks.items()}
        self._sign_chunks.clear()
        if self.file is not None:
            self.file.flush()

    def read(self, pair, start, stop, orbital_slices):
        first, second = orbital_slices
        if self.file is not None:
            return self.file[self._key(pair)][start:stop, first, second]
        offset = 0
        selected = []
        for array in self.arrays[pair]:
            end = offset + len(array)
            if start < end and stop > offset:
                selected.append(array[max(0, start-offset):min(len(array), stop-offset), first, second])
            offset = end
            if offset >= stop:
                break
        return selected[0] if len(selected) == 1 else np.concatenate(selected, axis=0)

    def close(self):
        if self.file is not None:
            self.file.close()


def _transform_pairs(mydf, mo, kpts, cache, workspace_bytes):
    nkpts, nao, nmo = mo.shape
    # Bound the complex AO buffer, transformed block, and unpacking workspace.
    rows = max(1, min(mydf.blockdim, int(workspace_bytes // (64 * (nao**2 + nmo**2)))))
    for pair in np.ndindex((nkpts, nkpts)):
        _, _, combined, orbital_slice = _conc_mos(mo[pair[0]], mo[pair[1]])
        for real, imag, sign in mydf.sr_loop(kpts[list(pair)], workspace_bytes / 1e6,
                                          False, rows):
            transformed = _ao2mo.r_e2(real + 1j * imag, combined, orbital_slice, [], None)
            cache.append(pair, transformed.reshape(-1, nmo, nmo), sign)
        if pair not in cache.naux:
            raise ValueError(f'No DF auxiliary factors for k-point pair {pair}')
    cache.finish()


def _transfer_groups(kconserv, cache):
    groups = {}
    for pair in np.ndindex(kconserv.shape[:2]):
        mapping = tuple(int(k) for k in kconserv[pair])
        groups.setdefault(mapping, []).append(pair)
    for mapping, left in groups.items():
        reference = cache.signs[left[0]]
        for pair in left + list(enumerate(mapping)):
            if not np.array_equal(cache.signs[pair], reference):
                raise ValueError('Incompatible DF auxiliary rows/signs for a conserving transfer')
    return groups


def _channel(cache, groups, output, shape, left_slice, right_slice, nkpts, workspace_bytes):
    nleft, nright = shape[0] * shape[1], shape[2] * shape[3]
    # Allow for an accumulator, GEMM result, orbital reshaping, and signed inputs.
    pair_batch = max(1, min(nkpts, int(np.sqrt(workspace_bytes / (64 * nleft * nright)))))
    for mapping, left_pairs in groups.items():
        signs = cache.signs[left_pairs[0]]
        for left_start in range(0, len(left_pairs), pair_batch):
            left = left_pairs[left_start:left_start + pair_batch]
            for right_start in range(0, nkpts, pair_batch):
                right_stop = min(nkpts, right_start + pair_batch)
                right = [(k, mapping[k]) for k in range(right_start, right_stop)]
                integral = np.zeros((len(left) * nleft, len(right) * nright), dtype=complex)
                aux_batch = max(1, int(workspace_bytes // (64 * (len(left)*nleft + len(right)*nright))))
                for start in range(0, len(signs), aux_batch):
                    stop = min(len(signs), start + aux_batch)
                    zij = np.concatenate([cache.read(pair, start, stop, left_slice).reshape(stop-start, nleft)
                                          for pair in left], axis=1)
                    zkl = np.concatenate([cache.read(pair, start, stop, right_slice).reshape(stop-start, nright)
                                          for pair in right], axis=1)
                    zkl *= signs[start:stop, None]
                    # Deliberately no conjugation: PySCF uses Zij.T @ Zkl.
                    integral += zij.T @ zkl
                integral /= nkpts
                blocks = integral.reshape(len(left), nleft, len(right), nright)
                blocks = blocks.transpose(0, 2, 1, 3).reshape(len(left), len(right), *shape)
                if output.dtype.kind != 'c':
                    blocks = blocks.real
                # Each write covers a contiguous run of k3, including on HDF5.
                for position, (ki, kj) in enumerate(left):
                    output[ki, kj, right_start:right_stop] = blocks[position]


def _core_terms(cache, nkpts, nmo, ncore, dtype, workspace_bytes):
    j_pc = np.zeros((nkpts, nmo, ncore), dtype=dtype)
    k_pc = np.zeros_like(j_pc)
    if ncore == 0:
        return j_pc, k_pc
    rows = max(1, int(workspace_bytes // (64 * nmo**2)))
    for k in range(nkpts):
        signs = cache.signs[k, k]
        for start in range(0, len(signs), rows):
            stop = min(len(signs), start + rows)
            z = cache.read((k, k), start, stop, (slice(None), slice(None)))
            sign = signs[start:stop]
            diagonal = np.diagonal(z, axis1=1, axis2=2)
            j = diagonal.T @ (diagonal[:, :ncore] * sign[:, None]) / nkpts
            pa, ap = z[:, :, :ncore], z[:, :ncore, :]
            exchange = np.einsum('Lpc,Lpc,L->pc', pa, pa, sign) / (3 * nkpts)
            exchange += 2 * np.einsum('Lpc,Lcp,L->pc', pa, ap, sign).conj() / (3 * nkpts)
            j_pc[k] += j if np.dtype(dtype).kind == 'c' else j.real
            k_pc[k] += exchange if np.dtype(dtype).kind == 'c' else exchange.real
    return j_pc, k_pc


def build_eris(kcasscf, mo_coeff, ncore, ncas, *, disk=False, level=1):
    """Build ppaa, papa, paap and core terms with a bounded factor workspace.

    Returns (eris, j_pc, k_pc, hcore). Outputs is a dict of arrays for
    direct mode, or an open HDF5 file with one dataset per channel for disk
    mode. The caller owns the returned HDF5 file. All factor files are closed.
    """
    mydf = kcasscf._scf.with_df
    kpts = np.asarray(kcasscf._scf.kpts)
    mo_coeff = np.asarray(mo_coeff)
    nkpts, nao, nmo = mo_coeff.shape
    nocc = ncore + ncas
    if len(kpts) != nkpts or not 0 <= ncore < nocc <= nmo:
        raise ValueError('Invalid k-point or active-space dimensions')
    dtype = np.float64 if gamma_point(kpts) and not np.iscomplexobj(mo_coeff) else np.complex128
    mo_kpts = np.asarray(mo_coeff, dtype=complex)
    active = slice(ncore, nocc)
    shapes = {'ppaa': (nmo, nmo, ncas, ncas), 
              'papa': (nmo, ncas, nmo, ncas),
              'paap': (nmo, ncas, ncas, nmo)}
    output_bytes = sum(nkpts**3 * int(np.prod(shape)) * np.dtype(dtype).itemsize 
                       for shape in shapes.values())
    available = (kcasscf.max_memory - lib.current_memory()[0]) * 1e6
    if not disk:
        available -= output_bytes
    # Keep a margin for hcore, Python objects, and BLAS buffers.
    workspace_bytes = max(1e6, available * 0.4)
    cache_mem = 0 if disk else max(0, available * 0.4) / 1e6
    cache = _PairCache(cache_mem, nmo)
    eris = lib.H5TmpFile() if disk else {}
    log = lib.logger.new_logger(kcasscf)
    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
    try:
        if mydf._cderi is None:
            mydf.build()
        _transform_pairs(mydf, mo_kpts, kpts, cache, workspace_bytes)
        groups = _transfer_groups(kpts_helper.get_kconserv(mydf.cell, kpts), cache)
        log.debug('DF pair cache: %s; batch workspace %.2f MB',
                  'HDF5' if cache.file is not None else 'RAM', workspace_bytes / 1e6)
        t1 = log.timer('density fitting ao2mo Lpq (shared pair cache)', *t1)
        for name, left, right in (
                ('ppaa', (slice(None), slice(None)), (active, active)),
                ('papa', (slice(None), active), (slice(None), active)),
                ('paap', (slice(None), active), (active, slice(None)))):
            full_shape = (nkpts,) * 3 + shapes[name]
            if disk:
                output = eris.create_dataset(name, shape=full_shape, dtype=dtype)
            else:
                output = eris[name] = np.empty(full_shape, dtype=dtype)
            _channel(cache, groups, output, shapes[name], left, right, nkpts, workspace_bytes)
            t1 = log.timer(f'density fitting ao2mo {name}', *t1)
        j_pc, k_pc = _core_terms(cache, nkpts, nmo, ncore, dtype, workspace_bytes) if level == 1 else (None, None)
        t1 = log.timer('density fitting ao2mo j_pc, k_pc', *t1)
        hcore = kcasscf.get_hcore()
        log.timer('hcore generation', *t1)
        if disk:
            eris.flush()
        return eris, j_pc, k_pc, hcore
    except BaseException:
        if disk:
            eris.close()
        raise
    finally:
        cache.close()


def build_cas_eris(mc, mo_cas_kpts, kconserv=None):
    """Build active Bloch ERIs, including the periodic 1/nkpts factor.

    Transform every DF pair once, retain signed auxiliary rows, and batch
    conserving pair products. The pair cache spills to HDF5 if needed.
    args:
        mc : KASSCF or KLASSCF object
        mo_cas_kpts : (nk,nao, nact)
    Returns:
        eris_kpts: (nk,nk,nk,nact,nact,nact,nact) array
    """
    mydf = mc._scf.with_df
    kpts = np.asarray(mc._scf.kpts)
    mo_cas_kpts = np.asarray(mo_cas_kpts)
    nkpts, nao, ncas = mo_cas_kpts.shape

    if len(kpts) != nkpts or ncas < 1:
        raise ValueError('Invalid k-point or active-space dimensions')
    
    dtype = np.float64 if gamma_point(kpts) and not np.iscomplexobj(mo_cas_kpts) else np.complex128
    output_bytes = nkpts ** 3 * ncas ** 4 * np.dtype(dtype).itemsize
    available = (mc.max_memory - lib.current_memory()[0]) * 1e6 - output_bytes
    workspace_bytes = max(1e6, available * 0.4)
    cache = _PairCache(max(0, available * 0.4) / 1e6, ncas)
    log = lib.logger.new_logger(mc)
    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
    try:
        if mydf._cderi is None:
            mydf.build()
        _transform_pairs(mydf, np.asarray(mo_cas_kpts, dtype=complex), kpts, cache, workspace_bytes)
        if kconserv is None:
            kconserv = kpts_helper.get_kconserv(mydf.cell, kpts)
        groups = _transfer_groups(kconserv, cache)
        t1 = log.timer('get_h2cas DF pair transforms', *t1)
        output = np.empty((nkpts,) * 3 + (ncas,) * 4, dtype=dtype)
        all_orbs = (slice(None), slice(None))
        _channel(cache, groups, output, (ncas,) * 4, all_orbs, all_orbs,
                 nkpts, workspace_bytes)
        log.timer('get_h2cas batched DF contraction', *t1)
        return output
    finally:
        cache.close()


def transform_eri_kpts_to_wannier(eri_kpts, mo_phase, kconserv):
    """Transform Bloch active-space ERIs to the Wannier basis.

    Args:
        eri_kpts : (nk,nk,nk,n,n,n,n) array, blocks (k1,k2,k3,a,b,c,d) with
            implicit k4 = kconserv[k1,k2,k3].
        mo_phase : (nk,n,nw) array U[k,a,w].
        kconserv : (nk,nk,nk) integer array.

    Returns:
        (nw,nw,nw,nw) array, no extra normalization:
        W[w,x,y,z] = sum_{k1,k2,k3,a,b,c,d} U*[k1,a,w] U[k2,b,x] U*[k3,c,y]
                     U[k4,d,z] eri_kpts[k1,k2,k3,a,b,c,d]
        i.e. einsum('Aaw,Bbx,Ccy,Ddz,ABCabcd->wxyz', U*, U, U*, U, eri) with
        D fixed by kconserv.

    Cost (N = nk*n): summing block by block is O(nk^7 n^5) (O(nk^7 n^8) for
    the brute-force sum). The transform is linear, so all blocks are first
    scattered into one dense N^4 tensor, then each axis is transformed once
    by a GEMM: O(N^5) = O(nk^5 n^5) time, O(N^4) memory.
    """
    blocks = np.asarray(eri_kpts)
    phase = np.asarray(mo_phase)
    nkpts, ncas, nactive = phase.shape

    _check_shape(blocks, (nkpts,)*3 + (ncas,)*4, label="eri_kpts")

    conservation = np.asarray(kconserv)
    _check_shape(conservation, (nkpts,)*3, label="kconserv")
    if not np.issubdtype(conservation.dtype, np.integer):
        raise ValueError('kconserv must be an integer array')
    
    if np.any(conservation < 0) or np.any(conservation >= nkpts):
        raise ValueError('kconserv contains an invalid momentum index')
    
    nblock = nkpts*ncas
    
    full = np.zeros((nblock,)*4, dtype=np.result_type(blocks, phase))
    
    view = full.reshape((nkpts,ncas)*4).transpose(0,2,4,6,1,3,5,7)
    
    k1, k2, k3 = np.ogrid[:nkpts, :nkpts, :nkpts]
    
    view[k1,k2,k3,conservation] = blocks
    
    matrix = phase.reshape(nblock, nactive)
    for axis, factor in enumerate((matrix.conj().T, matrix.T, matrix.conj().T, matrix.T)):
        full = np.moveaxis(np.tensordot(factor, full, axes=(1, axis)), 0, axis)
    
    return full
