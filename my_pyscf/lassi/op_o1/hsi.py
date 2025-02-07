import numpy as np
from scipy.sparse import linalg as sparse_linalg
from mrh.my_pyscf.lassi.op_o1.hams2ovlp import HamS2Ovlp, ham
from mrh.my_pyscf.lassi.citools import _fake_gen_contract_op_si_hdiag
from mrh.my_pyscf.lassi.op_o1.utilities import *
import functools

class HamS2OvlpOperators (HamS2Ovlp):
    __doc__ = HamS2Ovlp.__doc__ + '''

    SUBCLASS: Matrix-vector product

    Additional methods:
        get_ham_op, get_s2_op, get_ovlp_op
            Take no arguments and return LinearOperators of shape (nstates,nstates) which apply the
            respective operator to a SI trial vector.
        get_hdiag
            Take no arguments and return and ndarray of shape (nstates,) which contains the
            Hamiltonian diagonal
    '''
    def __init__(self, ints, nlas, hopping_index, lroots, h1, h2, mask_bra_space=None,
                 mask_ket_space=None, log=None, max_memory=2000, dtype=np.float64):
        HamS2Ovlp.__init__(self, ints, nlas, hopping_index, lroots, h1, h2,
                           mask_bra_space=mask_bra_space, mask_ket_space=mask_ket_space,
                           log=log, max_memory=max_memory, dtype=dtype)
        self.x = np.zeros (self.nstates, self.get_ci_dtype ())
        self.ox = np.zeros (self.nstates, self.dtype)

    def _umat_linequiv_(self, ifrag, iroot, umat, ivec, *args):
        if ivec==0:
            self.x = umat_dot_1frag_(self.x, umat.conj ().T, self.lroots, ifrag, iroot)
        elif ivec==1:
            self.ox = umat_dot_1frag_(self.ox, umat, self.lroots, ifrag, iroot)
        else:
            raise RuntimeError ("Invalid ivec = {}; must be 0 or 1".format (ivec))

    def _crunch_ox_env_(self, _crunch_fn, opid, *row): 
        if self._fn_row_has_spin (_crunch_fn):
            inv = row[2:-1]     
        else:
            inv = row[2:]
        self._prepare_spec_addr_ovlp_(row[0], row[1], *inv)
        data = _crunch_fn (*row)
        op = data[opid]
        ninv = data[2]
        op = self.canonical_operator_order (op, ninv)
        self._put_ox_(row[0], row[1], op, *inv)

    def _put_ox_(self, bra, ket, op, *inv):
        # TODO
        pass

    def _ham_op (self, x):
        self.x = x
        self.ox[:] = 0
        self._umat_linequiv_loop_(0) # U.conj () @ x
        for row in self.exc_1d: self._crunch_ox_env_(self._crunch_1d_, 0, *row)
        for row in self.exc_2d: self._crunch_ox_env_(self._crunch_2d_, 0, *row)
        for row in self.exc_1c: self._crunch_ox_env_(self._crunch_1c_, 0, *row)
        for row in self.exc_1c1d: self._crunch_ox_env_(self._crunch_1c1d_, 0, *row)
        for row in self.exc_1s: self._crunch_ox_env_(self._crunch_1s_, 0, *row)
        for row in self.exc_1s1c: self._crunch_ox_env_(self._crunch_1s1c_, 0, *row)
        for row in self.exc_2c: self._crunch_ox_env_(self._crunch_2c_, 0, *row)
        self._umat_linequiv_loop_(1) # U.T @ ox
        return self.ox

    def _s2_op (self, x):
        self.x = x
        self.ox[:] = 0
        self._umat_linequiv_loop_(0) # U.conj () @ x
        for row in self.exc_1d: self._crunch_ox_env_(self._crunch_1d_, 1, *row)
        for row in self.exc_2d: self._crunch_ox_env_(self._crunch_2d_, 1, *row)
        self._umat_linequiv_loop_(1) # U.T @ ox
        return self.ox

    def _ovlp_op (self, x):
        self.x = x
        self.ox[:] = 0
        self._umat_linequiv_loop_(0) # U.conj () @ x
        for bra, ket in self.exc_null:
            i0, i1 = self.offs_lroots[bra]
            j0, j1 = self.offs_lroots[ket]
            ovlp = self.crunch_ovlp (bra, ket)
            self.ox[i0:i1] = np.dot (ovlp, self.x[j0:j1])
            self.ox[j0:j1] = np.dot (ovlp.conj ().T, self.x[i0:i1])
        self._umat_linequiv_loop_(1) # U.T @ ox
        return self.ox

    def get_ham_op (self):
        return sparse_linalg.LinearOperator ([self.nstates,]*2, self.dtype, matvec=self._ham_op)

    def get_s2_op (self):
        return sparse_linalg.LinearOperator ([self.nstates,]*2, self.get_ci_dtype (),
                                             matvec=self._s2_op)

    def get_ovlp_op (self):
        return sparse_linalg.LinearOperator ([self.nstates,]*2, self.get_ci_dtype (),
                                             matvec=self._ovlp_op)

    def get_hdiag (self):
        # TODO
        pass

gen_contract_op_si_hdiag = functools.partial (_fake_gen_contract_op_si_hdiag, ham)

