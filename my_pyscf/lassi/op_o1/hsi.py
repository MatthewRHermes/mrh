import numpy as np
from scipy.sparse import linalg as sparse_linalg
from pyscf import lib
from pyscf.lib import logger
from mrh.my_pyscf.lassi import citools
from mrh.my_pyscf.lassi.op_o1 import frag
from mrh.my_pyscf.lassi.op_o1.rdm import LRRDM
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
        self.x = self.si = np.zeros (self.nstates, self.dtype)
        self.ox = np.zeros (self.nstates, self.dtype)
        self.ox1 = np.zeros (self.nstates, self.dtype)
        self.log.verbose = logger.DEBUG1
        self._init_cache_()

    def _init_cache_(self):
        t0 = (logger.process_clock (), logger.perf_counter ())
        self.init_cache_profiling ()
        self.excgroups_s = {}
        self.excgroups_h = {}
        for exc, fn in zip ((self.exc_1d, self.exc_2d, self.exc_1s),
                            (self._crunch_1d_, self._crunch_2d_, self._crunch_1s_)):
            self._groupexc_(exc, fn, has_s=True)
        for exc, fn in zip ((self.exc_1c, self.exc_1c1d, self.exc_1s1c, self.exc_2c),
                            (self._crunch_1c_, self._crunch_1c1d_, self._crunch_1s1c_, 
                             self._crunch_2c_)):
            self._groupexc_(exc, fn, has_s=False)
        self.log.debug1 (self.sprint_cache_profile ())
        self.log.timer_debug1 ('HamS2OvlpOperators operator cacheing', *t0)

    def _groupexc_(self, exc, fn, has_s=False):
        for row in exc:
            if self._fn_row_has_spin (fn):
                inv = row[2:-1]
            else:
                inv = row[2:]
            data = fn (*row)
            bra, ket = row[:2]
            row = inv.copy ()
            sinv = data[2]
            inv = list (set (inv))
            t0, w0 = logger.process_clock (), logger.perf_counter ()
            op = self.canonical_operator_order (data[0], sinv)
            opbralen = np.prod (self.lroots[inv,bra])
            opketlen = np.prod (self.lroots[inv,ket])
            op = op.reshape ((opbralen, opketlen), order='C')
            t1, w1 = logger.process_clock (), logger.perf_counter ()
            self.dt_oT += (t1-t0)
            self.dw_oT += (w1-w0)
            key = tuple (inv)
            val = self.excgroups_h.get (key, [])
            val.append ([op, bra, ket, row])
            self.excgroups_h[key] = val
            if has_s:
                t0, w0 = logger.process_clock (), logger.perf_counter ()
                op = self.canonical_operator_order (data[1], sinv)
                op = op.reshape ((opbralen, opketlen), order='C')
                t1, w1 = logger.process_clock (), logger.perf_counter ()
                self.dt_oT += (t1-t0)
                self.dw_oT += (w1-w0)
                val = self.excgroups_s.get (key, [])
                val.append ([op, bra, ket, row])
                self.excgroups_s[key] = val

    def init_cache_profiling (self):
        self.dt_1d, self.dw_1d = 0.0, 0.0
        self.dt_2d, self.dw_2d = 0.0, 0.0
        self.dt_1c, self.dw_1c = 0.0, 0.0
        self.dt_1c1d, self.dw_1c1d = 0.0, 0.0
        self.dt_1s, self.dw_1s = 0.0, 0.0
        self.dt_1s1c, self.dw_1s1c = 0.0, 0.0
        self.dt_2c, self.dw_2c = 0.0, 0.0
        self.dt_o, self.dw_o = 0.0, 0.0
        self.dt_u, self.dw_u = 0.0, 0.0
        self.dt_i, self.dw_i = 0.0, 0.0
        self.dt_oT, self.dw_oT = 0.0, 0.0

    def init_profiling (self):
        self.dt_gX, self.dw_gX = 0.0, 0.0
        self.dt_sX, self.dw_sX = 0.0, 0.0
        self.dt_oX, self.dw_oX = 0.0, 0.0
        self.dt_pX, self.dw_pX = 0.0, 0.0

    def sprint_cache_profile (self):
        fmt_str = '{:>5s} CPU: {:9.2f} ; wall: {:9.2f}'
        profile = fmt_str.format ('1d', self.dt_1d, self.dw_1d)
        profile += '\n' + fmt_str.format ('2d', self.dt_2d, self.dw_2d)
        profile += '\n' + fmt_str.format ('1c', self.dt_1c, self.dw_1c)
        profile += '\n' + fmt_str.format ('1c1d', self.dt_1c1d, self.dw_1c1d)
        profile += '\n' + fmt_str.format ('1s', self.dt_1s, self.dw_1s)
        profile += '\n' + fmt_str.format ('1s1c', self.dt_1s1c, self.dw_1s1c)
        profile += '\n' + fmt_str.format ('2c', self.dt_2c, self.dw_2c)
        profile += '\n' + fmt_str.format ('ovlp', self.dt_o, self.dw_o)
        profile += '\n' + fmt_str.format ('umat', self.dt_u, self.dw_u)
        profile += '\n' + fmt_str.format ('idx', self.dt_i, self.dw_i)
        profile += '\n' + fmt_str.format ('opT', self.dt_oT, self.dw_oT)
        return profile

    def sprint_profile (self):
        fmt_str = '{:>5s} CPU: {:9.2f} ; wall: {:9.2f}'
        profile = fmt_str.format ('getX', self.dt_gX, self.dw_gX)
        profile += '\n' + fmt_str.format ('olpX', self.dt_sX, self.dw_sX)
        profile += '\n' + fmt_str.format ('opX', self.dt_oX, self.dw_oX)
        profile += '\n' + fmt_str.format ('putX', self.dt_pX, self.dw_pX)
        return profile

    def get_single_rootspace_x (self, iroot):
        i, j = self.offs_lroots[iroot]
        return self.x[i:j]

    def get_xvec (self, iroot, *inv):
        fac = self.spin_shuffle[iroot] * self.fermion_frag_shuffle (iroot, inv)
        xvec = self.get_single_rootspace_x (iroot)
        return fac * xvec

    def put_oxvec_(self, vec, iroot, *inv):
        fac = self.spin_shuffle[iroot] * self.fermion_frag_shuffle (iroot, inv)
        i, j = self.offs_lroots[iroot]
        self.ox1[i:j] += fac * vec
        return

    def _umat_linequiv_(self, ifrag, iroot, umat, ivec, *args):
        if ivec==0:
            self.x = umat_dot_1frag_(self.x, umat.conj ().T, self.lroots, ifrag, iroot)
        elif ivec==1:
            self.ox = umat_dot_1frag_(self.ox, umat, self.lroots, ifrag, iroot)
        else:
            raise RuntimeError ("Invalid ivec = {}; must be 0 or 1".format (ivec))

    def _ham_op (self, x):
        t0 = (logger.process_clock (), logger.perf_counter ())
        self.init_profiling ()
        self.x[:] = x.flat[:]
        self.ox[:] = 0
        self._umat_linequiv_loop_(0) # U.conj () @ x
        for inv, group in self.excgroups_h.items (): self._opuniq_x_group_(inv, group)
        self._umat_linequiv_loop_(1) # U.T @ ox
        self.log.debug1 (self.sprint_profile ())
        self.log.timer_debug1 ('HamS2OvlpOperators._ham_op', *t0)
        return self.ox.copy ()

    def _s2_op (self, x):
        t0 = (logger.process_clock (), logger.perf_counter ())
        self.init_profiling ()
        self.x[:] = x.flat[:]
        self.ox[:] = 0
        self._umat_linequiv_loop_(0) # U.conj () @ x
        for inv, group in self.excgroups_s.items (): self._opuniq_x_group_(inv, group)
        self._umat_linequiv_loop_(1) # U.T @ ox
        self.log.debug1 (self.sprint_profile ())
        self.log.timer_debug1 ('HamS2OvlpOperators._s2_op', *t0)
        return self.ox.copy ()

    def _opuniq_x_group_(self, inv, group):
        '''All unique operations which have a set of nonspectator fragments in common'''
        self.ox1[:] = 0
        for op, bra, ket, myinv in group:
            self._opuniq_x_(op, bra, ket, *myinv)
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        for bra in range (self.nroots):
            i, j = self.offs_lroots[bra]
            self.ox[i:j] += transpose_sivec_with_slow_fragments (
                self.ox1[i:j], self.lroots[:,bra], *inv
            )[0]
        t1, w1 = logger.process_clock (), logger.perf_counter ()
        self.dt_pX += (t1-t0)
        self.dw_pX += (w1-w0)

    def _opuniq_x_(self, op, bra, ket, *inv):
        '''All operations which are unique in that a given set of fragment bra statelets are
        coupled to a given set of fragment ket statelets'''
        key = tuple ((bra, ket)) + inv
        inv = list (set (inv))
        tab = self.nonuniq_exc[key]
        bras, kets = self.nonuniq_exc[key].T
        self._op_x_(bras, kets, op, inv)
        self._op_x_(kets, bras, op.conj ().T, inv)
        return 

    def _op_x_(self, bras, kets, op, inv):
        t0, w0 = logger.process_clock (), logger.perf_counter ()

        ketvecs = {ket: self.get_xvec (ket, *inv) for ket in set (kets)}

        t1, w1 = logger.process_clock (), logger.perf_counter ()
        self.dt_gX += (t1-t0)
        self.dw_gX += (w1-w0)

        bravecs = {bra: 0.0 for bra in set (bras)}
        for bra, ket in zip (bras, kets):
            bravecs[bra] += self.ox_ovlp_part (bra, ket, ketvecs[ket], inv)

        t2, w2 = logger.process_clock (), logger.perf_counter ()
        self.dt_sX += (t2-t1)
        self.dw_sX += (w2-w1)

        bravecs = {bra: lib.dot (op, vec.T) for bra, vec in bravecs.items ()}

        t3, w3 = logger.process_clock (), logger.perf_counter ()
        self.dt_oX += (t3-t2)
        self.dw_oX += (w3-w2)

        for bra, vec in bravecs.items ():
            self.put_oxvec_(vec.ravel (), bra, *inv)

        t4, w4 = logger.process_clock (), logger.perf_counter ()
        self.dt_pX += (t4-t3)
        self.dw_pX += (w4-w3)

    def ox_ovlp_part (self, bra, ket, vec, inv):
        if (bra==ket): vec = vec * 0.5
        spec = np.ones (self.nfrags, dtype=bool)
        spec[inv] = False
        lr = 1
        for i in range (self.nfrags):
            lket = self.lroots[i,ket]
            if spec[i]:
                vec = vec.reshape (-1,lket,lr)
                o = self.ints[i].get_ovlp (bra, ket)
                vec = lib.einsum ('pq,lqr->plr', o, vec)
            else:
                lr = lr * lket
        return vec.reshape (-1,lr)

    def _ovlp_op (self, x):
        t0 = (logger.process_clock (), logger.perf_counter ())
        self.x[:] = x.flat[:]
        self.ox[:] = 0
        self._umat_linequiv_loop_(0) # U.conj () @ x
        for bra, ket in self.exc_null:
            i0, i1 = self.offs_lroots[bra]
            j0, j1 = self.offs_lroots[ket]
            ovlp = self.crunch_ovlp (bra, ket)
            self.ox[i0:i1] += np.dot (ovlp, self.x[j0:j1])
            self.ox[j0:j1] += np.dot (ovlp.conj ().T, self.x[i0:i1])
        self._umat_linequiv_loop_(1) # U.T @ ox
        self.log.timer_debug1 ('HamS2OvlpOperators._ovlp_op', *t0)
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
        t0 = (logger.process_clock (), logger.perf_counter ())
        self.ox[:] = 0
        for row in self.exc_1d: self._crunch_hdiag_env_(self._crunch_1d_, *row)
        for row in self.exc_2d: self._crunch_hdiag_env_(self._crunch_2d_, *row)
        self.log.timer_debug1 ('HamS2OvlpOperators.get_hdiag', *t0)
        return self.ox.copy ()

    def _crunch_hdiag_env_(self, _crunch_fn, *row): 
        if row[0] != row[1]: return
        if self._fn_row_has_spin (_crunch_fn):
            inv = row[2:-1]     
        else:
            inv = row[2:]
        self._prepare_spec_addr_ovlp_(row[0], row[1], *inv)
        ham, s2, sinv = _crunch_fn (*row)
        sinv = self.inv_unique (sinv)[::-1]
        key = tuple ((row[0], row[1])) + inv
        for bra, ket in self.nonuniq_exc[key]:
            if bra != ket: continue
            hdiag_nonspec = self.get_hdiag_nonspectator (ham, bra, *sinv)
            hdiag_spec = self.hdiag_spectator_ovlp (bra, *sinv)
            hdiag = np.multiply.outer (hdiag_nonspec, hdiag_spec)
            hdiag = transpose_sivec_with_slow_fragments (hdiag.ravel (), self.lroots[:,bra], *sinv)
            i, j = self.offs_lroots[bra] 
            self.ox[i:j] += hdiag.ravel ()
        #ham = self.canonical_operator_order (ham, ninv)
        #self._put_hdiag_(row[0], row[1], ham, *inv)

    def get_hdiag_nonspectator (self, ham, bra, *inv):
        for i in inv:
            n = self.lroots[i,bra]
            umat = self.ints[i].umat_root.get (bra, np.eye (n))
            umat = (umat[None,:,:] * umat[:,None,:]).reshape (n*n, n)
            umat = np.ascontiguousarray (umat.T)
            ham = ham.reshape (-1, n*n)
            ham = np.dot (umat, ham.T)
        return ham

    def hdiag_spectator_ovlp (self, rbra, *inv):
        fac = self.spin_shuffle[rbra] * self.spin_shuffle[rbra]
        fac *= self.fermion_frag_shuffle (rbra, inv)
        fac *= self.fermion_frag_shuffle (rbra, inv)
        spec = np.ones (self.nfrags, dtype=bool)
        for i in inv: spec[i] = False
        spec = np.where (spec)[0]
        specints = [self.ints[i] for i in spec]
        o = fac * np.ones ((1,1), dtype=self.dtype)
        for i in specints:
            o = np.multiply.outer (i.get_ovlp_inpbasis (rbra, rbra).diagonal (), o)
            o = o.ravel ()
        return o

    def _put_hdiag_(self, bra, ket, op, *inv):
        bra_rng = self._get_addr_range (bra, *inv) # profiled as idx
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        op_nbra = np.prod (op.shape[:op.ndim//2])
        op_nket = np.prod (op.shape[op.ndim//2:])
        op = op.reshape (op_nbra, op_nket).diagonal ()
        op = op + op.conj ()
        for ix, bra1 in enumerate (bra_rng):
            bra2, ket2, wgt = self._get_spec_addr_ovlp (bra1, bra1, *inv)
            idx = (bra2==ket2)
            t1, w1 = logger.process_clock (), logger.perf_counter ()
            self.ox[bra2[idx]] += wgt[idx] * op[ix]
            dt, dw = logger.process_clock () - t1, logger.perf_counter () - w1
            self.dt_s, self.dw_s = self.dt_s + dt, self.dw_s + dw
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        #self.dt_p, self.dw_p = self.dt_p + dt, self.dw_p + dw

#gen_contract_op_si_hdiag = functools.partial (_fake_gen_contract_op_si_hdiag, ham)
def gen_contract_op_si_hdiag (las, h1, h2, ci, nelec_frs, soc=0, nlas=None,
                              _HamS2Ovlp_class=HamS2OvlpOperators, _return_int=False, **kwargs):
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
        _return_int : logical
            If True, return the main intermediate object instead of the
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

    if _return_int: return outerprod

    ham_op = outerprod.get_ham_op ()
    s2_op = outerprod.get_s2_op ()
    ovlp_op = outerprod.get_ovlp_op ()
    hdiag = outerprod.get_hdiag ()
    raw2orth = citools.get_orth_basis (ci, las.ncas_sub, nelec_frs,
                                       _get_ovlp=outerprod.get_ovlp)
    return ham_op, s2_op, ovlp_op, hdiag, raw2orth

