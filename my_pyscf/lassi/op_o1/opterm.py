import numpy as np
from pyscf import lib

class OpTermGroup:
    '''A set of operators which address the same set of nonspectator fragments'''
    def __init__(self, inv):
        self.inv = inv
        self.ops = []
        self.ovlplink = None

    def append (self, val):
        self.ops.append (val)

    def neutral_only (self):
        new_ops = []
        for op in self.ops:
            neutral = True
            bra, ket = op.spincase_keys[0][:2]
            for inti in op.ints:
                if sum (inti.nelec_r[bra]) - sum (inti.nelec_r[ket]) > 0:
                    neutral = False
                    break
            if neutral:
                new_ops.append (op)
        if len (new_ops) == 0: return None
        new_group = OpTermGroup (self.inv)
        new_group.ops = new_ops
        new_group.ovlplink = self.ovlplink
        return new_group

    def subspace (self, keys):
        # Project into a subspace
        new_ops = []
        for op in self.ops:
            new_spincase_keys = []
            for spincase_key in op.spincase_keys:
                if spincase_key in keys:
                    new_spincase_keys.append (spincase_key)
            if len (new_spincase_keys) > 0:
                new_op = op.copy ()
                new_op.spincase_keys = new_spincase_keys
                new_ops.append (new_op)
        if len (new_ops) == 0: return None
        new_group = OpTermGroup (self.inv)
        new_group.ops = new_ops
        return new_group            

class OpTermBase:
    '''Elements of spincase_keys index nonuniq_exc to look up bras and kets addressed by this
    operator corresponding to a particular set of ket spin polarization quantum numbers.'''
    def __init__(self):
        self.spincase_keys = []

    def spincase_fprint (self, i):
        bra, ket = self.spincase_keys[i][:2]
        brastr = [inti.uroot_idx[bra] for inti in self.ints]
        ketstr = [inti.uroot_idx[ket] for inti in self.ints]
        return brastr, ketstr

    def spincase_mstrs (self, key):
        bra, ket = key[:2]
        brastr = tuple ([inti.spins_r[bra] for inti in self.ints])
        ketstr = tuple ([inti.spins_r[ket] for inti in self.ints])
        return brastr, ketstr

    def fprint (self):
        brastr, ketstr = self.spincase_fprint (0)
        brastr = [self.ints[i].spman[bra] for i, bra in enumerate (brastr)]
        ketstr = [self.ints[i].spman[ket] for i, ket in enumerate (ketstr)]
        return brastr, ketstr

    def get_inv_frags (self):
        return [inti.idx_frag for inti in self.ints]

    def get_formal_shape (self):
        return self.shape

class OpTermReducible (OpTermBase):
    def copy (self):
        return lib.view (self, self.__class__)

class OpTerm (OpTermReducible):
    def __init__(self, arr, ints, comp, _already_stacked=False):
        self.ints = ints
        self.comp = comp
        if (comp is None) or _already_stacked:
            self.arr = arr
        else:
            self.arr = np.stack (arr, axis=-1)
        super().__init__()

    def reduce_spin (self, bra, ket):
        if any ([i.smult_r[ket] is None for i in self.ints]):
            arr = self.arr
            if self.comp is not None:
                arr = arr.sum (-1)
            return arr.view (OpTermContracted)
        if self.comp is None:
            arr = self.arr.copy ()
        else:
            fac = np.ones (len (self.comp), dtype=float)
            for i, comp_i in enumerate (self.comp):
                for intj, comp_ij in zip (self.ints, comp_i):
                    fac[i] *= intj.spin_factor_component (bra, ket, comp_ij)
            arr = np.dot (self.arr, fac)
        fac = np.prod ([inti.spin_factor_constant (bra, ket) for inti in self.ints])
        arr *= fac
        return arr.view (OpTermContracted)

    def reduce_spin_sum (self, facs):
        arrs = np.stack ([self.reduce_spin (key[0], key[1]) for key in self.spincase_keys], axis=-1)
        return np.dot (arrs, facs)

    @property
    def ndim (self):
        ncomp = 1 - int (self.comp is None)
        return self.arr.ndim-ncomp

    def transpose (self, *idx):
        if self.comp is not None:
            idx += (self.arr.ndim-1,)
        arr = self.arr.transpose (*idx)
        return OpTerm (arr, self.ints, self.comp, _already_stacked=True)

    @property
    def shape (self):
        if self.comp is None:
            return self.arr.shape
        else:
            return self.arr.shape[:-1]

    @property
    def size (self):
        return np.prod (self.shape)

    def reshape (self, new_shape, order='C'):
        assert (order.upper () == 'C')
        # TODO: generalize order
        if self.comp is not None:
            new_shape += (self.arr.shape[-1],)
        arr = self.arr.reshape (new_shape, order=order)
        return OpTerm (arr, self.ints, self.comp, _already_stacked=True)

    def maxabs (self):
        return np.amax (np.abs (self.arr))

def as_opterm (op):
    if isinstance (op, OpTermBase):
        return op
    else:
        return op.view (OpTermContracted)

def reduce_spin (op, bra, ket):
    if isinstance (op, OpTermReducible):
        return op.reduce_spin (bra, ket)
    else:
        return op

class OpTermContracted (np.ndarray, OpTermBase):
    ''' Just farm the dot method to pyscf.lib.dot '''
    def dot (self, other):
        return lib.dot (self, other)

    def maxabs (self):
        return np.amax (np.abs (self))

class OpTermNFragments (OpTermReducible):
    def __init__(self, op, idx, d, ints, do_crunch=True):
        assert (len (idx) == len (d))
        isort = np.argsort (idx)
        if do_crunch and (op.ndim == len (isort)):
            self.op = op.transpose (isort)
        else:
            self.op = op
        self.idx = [idx[i] for i in isort]
        self.d = [d[i] for i in isort]
        self.ints = [ints[i] for i in isort]
        self.lroots_bra = [d.shape[0] for d in self.d]
        self.lroots_ket = [d.shape[1] for d in self.d]
        self.norb = [d.shape[2] for d in self.d]
        self.comp = None
        if do_crunch: self._crunch_()
        super().__init__()

    def get_formal_shape (self):
        return np.prod (self.lroots_bra), np.prod (self.lroots_ket)

    def reshape (self, new_shape, **kwargs):
        pass

    def _crunch_(self):
        raise NotImplementedError

    def conj (self, do_crunch=False):
        d = [d.conj () for d in self.d]
        op = self.op.conj ()
        return self.__class__(op, self.idx, d, self.ints, do_crunch=do_crunch)

    def transpose (self, do_crunch=False):
        d = [d.transpose (1,0,2) for d in self.d]
        op = self.op.transpose (*self.op_transpose_axes)
        return self.__class__(op, self.idx, d, self.ints, do_crunch=do_crunch)

    @property
    def T (self): return self.transpose ()

    def get_size (self):
        # d should not be copies, but op is
        return self.op.size

    def reduce_spin (self, bra, ket, do_crunch=False):
        if any ([i.smult_r[ket] is None for i in self.ints]):
            return self
        fac = [inti.spin_factor_constant (bra, ket) for inti in self.ints]
        d = [self.d[i].copy () * fac[i] for i in range (len (fac))]
        if do_crunch:
            op = self.op.copy ()
        else:
            op = self.reduce_spin_op (bra, ket, fac)
        return self.__class__(op, self.idx, d, self.ints, do_crunch=do_crunch)

    def reduce_spin_op (self, bra, ket, fac):
        raise NotImplementedError

    def reduce_spin_sum (self, facs):
        raise NotImplementedError ('I need to be less tired to figure out how to make this general')

    @property
    def size (self): return self.get_size ()

    @property
    def op_transpose_axes (self): return list (range (self.op.ndim))

    def maxabs (self):
        return np.amin ([np.amax (np.abs (d)) for d in self.d] + [np.amax (np.abs (self.op)),])

# Notes on how to factor this:
# in the three passes, the fingerprints identifying meaningful distinct (bra,ket) tuples are
#
#   (inner loop over mkp, mkq and multiply by spin factor on lookup)
#   bp bq ** ** (inv = p,q,r,s on vector lookup)
#   kp kq kr ks (inv = r,s on vector storage)
#
#   (inner loop over mbr and multiply by spin factor on storage)
#   bp bq br ** (inv = r,s on vector lookup)
#   ** ** kr ks (inv = s on vector storage)
#
#   (inner loop over mbs and multiply by spin factor on storage)
#   bp bq br bs (inv = s on vector lookup)
#   ** ** ** ks (final pass: put into ox)
# 
# The "inner loops" are achieved by indexing down spincase_keys to only couplings distinct in
# the relevant fragments (which are different in the three passes)
# There are only four m components overall (aaaa, abba, baab, and bbbb).
# These components are currently different OpTerms, not just different spincases. To refactor this
# I need to pin together OpTerms. I'm not sure whether all the bras and kets of the different
# OpTerms are shared.
class OpTerm4Fragments (OpTermNFragments):
    def _crunch_(self):
        self.op = lib.einsum ('aip,bjq,pqrs->rsbaji', self.d[0], self.d[1], self.op)
        self.op = np.ascontiguousarray (self.op)

    def dot (self, other):
        ncol = other.shape[1]
        shape = [ncol,] + self.lroots_ket[::-1]
        t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
        other = other.T.reshape (*shape)
        t1 = (lib.logger.process_clock (), lib.logger.perf_counter ())
        ox = lib.einsum ('rsbaji,zlkji->rsbazlk', self.op, other)
        t2 = (lib.logger.process_clock (), lib.logger.perf_counter ())
        ox = lib.einsum ('ckr,rsbazlk->scbazl', self.d[2], ox)
        t3 = (lib.logger.process_clock (), lib.logger.perf_counter ())
        ox = lib.einsum ('dls,scbazl->dcbaz', self.d[3], ox)
        t4 = (lib.logger.process_clock (), lib.logger.perf_counter ())
        ox = ox.ravel ().reshape (np.prod (self.lroots_bra), ncol)
        t5 = (lib.logger.process_clock (), lib.logger.perf_counter ())
        self.dt_4fr = t1[0] - t0[0]
        self.dw_4fr = t1[1] - t0[1]
        self.dt_4f1 = t2[0] - t1[0]
        self.dw_4f1 = t2[1] - t1[1]
        self.dt_4f2 = t3[0] - t2[0]
        self.dw_4f2 = t3[1] - t2[1]
        self.dt_4f3 = t4[0] - t3[0]
        self.dw_4f3 = t4[1] - t3[1]
        self.dt_4fr += t5[0] - t4[0]
        self.dw_4fr += t5[1] - t4[1]
        return ox

    op_transpose_axes = (0,1,4,5,2,3)

    def reduce_spin_op (self, bra, ket, fac):
        fac = fac[0] * fac[1]
        return self.op.copy () * fac

    def reduce_spin_sum (self, facs):
        if any ([i.smult_r[self.spincase_keys[0][1]] is None for i in self.ints]):
            myfac = 1
        else:
            myfac = np.dot (facs, [np.prod ([inti.spin_factor_constant (key[0], key[1])
                                             for inti in self.ints])
                                   for key in self.spincase_keys])
        return self.__class__(self.op.copy () * myfac, self.idx, self.d, self.ints, do_crunch=False)

    def fdm_dot (self, fdm):
        output_shape = fdm.shape[:-1]
        ncols = fdm.shape[-1]
        nrows = np.prod (output_shape)
        dot_shape = [nrows,] + self.lroots_bra[::-1] + self.lroots_ket[::-1]
        arr = fdm.reshape (*dot_shape)
        arr = lib.einsum ('zdcbalkji,ckr,dls->zrsbaji', arr, self.d[2], self.d[3])
        arr = lib.einsum ('zrsbaji,rsbaji->z', arr, self.op)
        return arr.reshape (*output_shape)

def fdm_dot (fdm, op1):
    if callable (getattr (op1, 'fdm_dot', None)):
        return op1.fdm_dot (fdm)
    else:
        return np.dot (fdm, op1.ravel ())


