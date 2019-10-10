from pyscf import lib
import numpy as np
from scipy import linalg
from mrh.lib.helper import load_library
import ctypes
libsint = load_library ('libsint')

class sparsedf_array (np.ndarray):
    def __new__(cls, inp, nmo=None):
        assert (inp.flags['C_CONTIGUOUS'] or inp.flags['F_CONTIGUOUS'])
        self = np.asarray (inp).view (cls)
        self.naux = self.shape[0]
        self.nmo = nmo
        if self.nmo is None:
            if self.ndim == 3:
                nmo1 = 1
                for self.nmo1 in range (1, self.shape[1]):
                    off += nmo1
                    if off == self.shape[1]: break
                    nmo1 += 1
                self.nmo = (nmo1, nmo1)
            else:
                self.nmo = self.shape[1:]
        return self

    def __array_finalize__(self, obj):
        if obj is None: return
        self.naux = getattr(obj, 'naux')
        self.nmo = getattr(obj, 'nmo', None)

    def pack_mo (self):
        if self.ndim == 2: return self
        elif self.ndim == 3: return sparsedf_array (lib.pack_tril (self), nmo=self.nmo)
        else: raise (RuntimeError, 'wrong number of dimensions')

    def unpack_mo (self):
        if self.ndim == 3: return self
        elif self.ndim == 2: return sparsedf_array (lib.unpack_tril (self), nmo=self.nmo)
        else: raise (RuntimeError, 'wrong number of dimensions')

    def transpose_mo (self):
        assert (self.ndim == 3), "Can't transpose packed lower-triangular matrix (by definition this is pointless)"
        return sparsedf_array (lib.numpy_helper.transpose (self, axes=(0,2,1), inplace=(self.nmo[0] == self.nmo[1])), nmo=(self.nmo[1], self.nmo[0]))
        
    def naux_fast (self): # Since naux is always the first index, this corresponds to making the array F-contiguous
        return sparsedf_array (np.asfortranarray (self), nmo=self.nmo)

    def naux_slow (self): # Since naux is always the first index, this corresponds to making the array C-contiguous
        return sparsedf_array (np.ascontiguousnarray (self), nmo=self.nmo)

    def get_sparsity_ (self, thresh=1e-8):
        metric = linalg.norm (self, axis=0)
        if metric.ndim = 1: metric = lib.unpack_tril (metric)
        metric = metric > thresh
        self.iao_nent = np.count_nonzero (metric, axis=0).astype (np.int32)
        self.nent_max = np.amax (self.iao_nent)
        self.iao_entlist = -np.ones ((self.nmo[0], self.nent_max), dtype=np.int32)
        for irow, nent in enumerate (self.iao_nent):
            self.iao_entlist[irow,:nent] = np.where (metric[irow])[0]
        self.iao_sort = np.argsort (self.iao_nent).astype (np.int32)

    def contract1 (self, cmat):
        if self.ndim == 3: return self.pack_mo ()
        if not self.flags.['C_CONTIGUOUS']: self = self.naux_slow ()
        nao = self.nmo[0]
        nmo = cmat.shape[1]
        if self.nent_max is None: self.get_sparsity_ ()
        prod = np.zeros ((nao, nmo, self.naux), dtype=self.dtype).view (sparsedf_array)
        wrk1 = np.zeros ((lib.num_threads (), self.nent_max, self.naux), dtype = self.dtype).view (sparsedf_array)
        wrk2 = np.zeros ((lib.num_threads (), self.nent_max, self.nmo), dtype = self.dtype).view (sparsedf_array)
        libsint.SINT_SDCDERI_DDMAT (self.cytpes.data_as (ctypes.c_void_p),
            cmat.ctypes.data_as (ctypes.c_void_p),
            prod.ctypes.data_as (ctypes.c_void_p),
            wrk1.ctypes.data_as (ctypes.c_void_p),
            wrk2.ctypes.data_as (ctypes.c_void_p), 
            self.iao_sort.ctypes.data_as (ctypes.c_void_p),
            self.iao_nent.ctypes.data_as (ctypes.c_void_p),
            self.iao_entlist.ctypes.data_as (ctypes.c_void_p),
            ctypes.c_int (nao), ctypes.c_int (naux),
            ctypes.c_int (nmo), ctypes.c_int (nent_max))
        wrk1 = wrk2 = None
        return prod
            

)

