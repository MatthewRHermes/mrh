import numpy as np
from scipy.sparse import linalg as sparse_linalg
from pyscf import lib
from pyscf.lib import logger
from mrh.my_pyscf.lassi import citools
from mrh.my_pyscf.lassi.op_o1 import frag
from mrh.my_pyscf.lassi.op_o1.hams2ovlp import HamS2Ovlp, ham, soc_context
from mrh.my_pyscf.lassi.citools import _fake_gen_contract_op_si_hdiag
from mrh.my_pyscf.lassi.op_o1.utilities import *
import functools
from itertools import product

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
        self.x = np.zeros (self.nstates, self.dtype)
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
        # TODO: transpose the nested loops and vectorize the ix, (bra1, ket1) loop
        bra_rng = self._get_addr_range (bra, *inv) # profiled as idx
        ket_rng = self._get_addr_range (ket, *inv) # profiled as idx
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        op = op.flat
        for ix, (bra1, ket1) in enumerate (product (bra_rng, ket_rng)):
            bra2, ket2, wgt = self._get_spec_addr_ovlp (bra1, ket1, *inv)
            t1, w1 = logger.process_clock (), logger.perf_counter ()
            for bra3, ket3, w in zip (bra2, ket2, wgt):
                self.ox[bra3] += w * op[ix] * self.x[ket3]
                self.ox[ket3] += w * op[ix].conj () * self.x[bra3]
            dt, dw = logger.process_clock () - t1, logger.perf_counter () - w1
            self.dt_s, self.dw_s = self.dt_s + dt, self.dw_s + dw
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_p, self.dw_p = self.dt_p + dt, self.dw_p + dw

    def _ham_op (self, x):
        self.x[:] = x[:]
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
        return self.ox.copy ()

    def _s2_op (self, x):
        self.x[:] = x[:]
        self.ox[:] = 0
        self._umat_linequiv_loop_(0) # U.conj () @ x
        for row in self.exc_1d: self._crunch_ox_env_(self._crunch_1d_, 1, *row)
        for row in self.exc_2d: self._crunch_ox_env_(self._crunch_2d_, 1, *row)
        for row in self.exc_1s: self._crunch_ox_env_(self._crunch_1s_, 1, *row)
        self._umat_linequiv_loop_(1) # U.T @ ox
        return self.ox.copy ()

    def _ovlp_op (self, x):
        self.x[:] = x[:]
        self.ox[:] = 0
        self._umat_linequiv_loop_(0) # U.conj () @ x
        for bra, ket in self.exc_null:
            i0, i1 = self.offs_lroots[bra]
            j0, j1 = self.offs_lroots[ket]
            ovlp = self.crunch_ovlp (bra, ket)
            self.ox[i0:i1] += np.dot (ovlp, self.x[j0:j1])
            self.ox[j0:j1] += np.dot (ovlp.conj ().T, self.x[i0:i1])
        self._umat_linequiv_loop_(1) # U.T @ ox
        return self.ox.copy ()

    def get_ham_op (self):
        return sparse_linalg.LinearOperator ([self.nstates,]*2, dtype=self.dtype,
                                             matvec=self._ham_op)

    def get_s2_op (self):
        return sparse_linalg.LinearOperator ([self.nstates,]*2, dtype=self.dtype,
                                             matvec=self._s2_op)

    def get_ovlp_op (self):
        return sparse_linalg.LinearOperator ([self.nstates,]*2, dtype=self.dtype,
                                             matvec=self._ovlp_op)

    def get_hdiag (self):
        # TODO: make this not stupid
        ham_op = self.get_ham_op ()
        hdiag = np.zeros_like (self.ox)
        x = np.zeros_like (self.x)
        for i in range (self.nstates):
            x[:] = 0
            x[i] = 1.0
            hdiag[i] = ham_op (x)[i]
        return hdiag

#gen_contract_op_si_hdiag = functools.partial (_fake_gen_contract_op_si_hdiag, ham)
def gen_contract_op_si_hdiag (las, h1, h2, ci, nelec_frs, soc=0, nlas=None,
                              _HamS2Ovlp_class=HamS2OvlpOperators, _do_kernel=True, **kwargs):
    ''' Build Hamiltonian, spin-squared, and overlap matrices in LAS product state basis

    Args:
        las : instance of :class:`LASCINoSymm`
        h1 : ndarray of size ncas**2
            Contains effective 1-electron Hamiltonian amplitudes in second quantization
        h2 : ndarray of size ncas**4
            Contains 2-electron Hamiltonian amplitudes in second quantization
        ci : list of list of ndarrays
            Contains all CI vectors
        nelec_frs : ndarray of shape (nfrags,nroots,2)
            Number of electrons of each spin in each rootspace in each
            fragment

    Kwargs:
        soc : integer
            Order of spin-orbit coupling included in the Hamiltonian
        nlas : sequence of length (nfrags)
            Number of orbitals in each fragment
        _HamS2Ovlp_class : class
            The main intermediate class
        _do_kernel : logical
            If false, return the main intermediate object before running kernel, instead of the
            operator matrices
        
    Returns: 
        ham_op : LinearOperator of shape (nstates,nstates)
            Hamiltonian in LAS product state basis
        s2_op : LinearOperator of shape (nstates,nstates)
            Spin-squared operator in LAS product state basis
        ovlp_op : LinearOperator of shape (nstates,nstates)
            Overlap matrix of LAS product states 
        hdiag : ndarray of shape (nstates,)
            Diagonal element of Hamiltonian matrix
        raw2orth : LinearOperator of shape (nstates_orth, nstates)
            Projects SI vector columns into an orthonormal basis,
            eliminating linear dependencies (nstates_orth <= nstates)
    '''
    log = lib.logger.new_logger (las, las.verbose)
    if nlas is None: nlas = las.ncas_sub
    max_memory = getattr (las, 'max_memory', las.mol.max_memory)
    dtype = h1.dtype
    nfrags, nroots = nelec_frs.shape[:2]
    if soc>1: raise NotImplementedError ("Spin-orbit coupling of second order")

    # Handle possible SOC
    spin_pure, h1, h2, ci, nelec_frs, nlas, spin_shuffle_fac = soc_context (
        h1, h2, ci, nelec_frs, soc, nlas)

    # First pass: single-fragment intermediates
    hopping_index, ints, lroots = frag.make_ints (las, ci, nelec_frs, nlas=nlas)
    nstates = np.sum (np.prod (lroots, axis=0))

    # Second pass: upper-triangle
    t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
    outerprod = _HamS2Ovlp_class (ints, nlas, hopping_index, lroots, h1, h2, dtype=dtype,
                                     max_memory=max_memory, log=log)
    if soc and not spin_pure:
        outerprod.spin_shuffle = spin_shuffle_fac
    lib.logger.timer (las, 'LASSI Hamiltonian second intermediate indexing setup', *t0)
    if not _do_kernel: return outerprod

    ham_op = outerprod.get_ham_op ()
    s2_op = outerprod.get_s2_op ()
    ovlp_op = outerprod.get_ovlp_op ()
    hdiag = outerprod.get_hdiag ()
    raw2orth = citools.get_orth_basis (ci, las.ncas_sub, nelec_frs,
                                       _get_ovlp=outerprod.get_ovlp)
    return ham_op, s2_op, ovlp_op, hdiag, raw2orth

