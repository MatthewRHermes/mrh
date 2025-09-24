import numpy as np
from pyscf import lib

class OpTermBase: pass

class OpTermContracted (np.ndarray, OpTermBase):
    ''' Just farm the dot method to pyscf.lib.dot '''
    def dot (self, other):
        return lib.dot (self, other)

class OpTermNFragments (OpTermBase):
    def __init__(self, op, idx, d, do_crunch=True):
        assert (len (idx) == len (d))
        isort = np.argsort (idx)
        if do_crunch and (op.ndim == len (isort)):
            self.op = op.transpose (isort)
        else:
            self.op = op
        self.idx = [idx[i] for i in isort]
        self.d = [d[i] for i in isort]
        self.lroots_bra = [d.shape[0] for d in self.d]
        self.lroots_ket = [d.shape[1] for d in self.d]
        self.norb = [d.shape[2] for d in self.d]
        if do_crunch: self._crunch_()

    def reshape (self, new_shape, **kwargs):
        pass

    def _crunch_(self):
        raise NotImplementedError

    def conj (self, do_crunch=False):
        d = [d.conj () for d in self.d]
        op = self.op.conj ()
        return self.__class__(op, self.idx, d, do_crunch=do_crunch)

    def transpose (self, do_crunch=False):
        d = [d.transpose (1,0,2) for d in self.d]
        op = self.op.transpose (*self.op_transpose_axes)
        return self.__class__(op, self.idx, d, do_crunch=do_crunch)

    @property
    def T (self): return self.transpose ()

    def get_size (self):
        # d should not be copies, but op is
        return self.op.size

    @property
    def size (self): return self.get_size ()

    @property
    def op_transpose_axes (self): return list (range (self.op.ndim))

class OpTerm4Fragments (OpTermNFragments):
    def _crunch_(self):
        self.op = lib.einsum ('aip,bjq,pqrs->rsbaji', self.d[0], self.d[1], self.op)
        self.op = np.ascontiguousarray (self.op)

    def dot (self, other):
        ncol = other.shape[1]
        shape = [ncol,] + self.lroots_ket[::-1]
        other = other.T.reshape (*shape)
        ox = lib.einsum ('rsbaji,zlkji->rsbazlk', self.op, other)
        ox = lib.einsum ('ckr,rsbazlk->scbazl', self.d[2], ox)
        ox = lib.einsum ('dls,scbazl->dcbaz', self.d[3], ox)
        ox = ox.reshape (np.prod (self.lroots_bra), ncol)
        return ox

    op_transpose_axes = (0,1,4,5,2,3)

