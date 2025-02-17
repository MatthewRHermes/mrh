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
        self.log.verbose = logger.DEBUG1

    def get_single_rootspace_x (self, iroot):
        i, j = self.offs_lroots[iroot]
        return self.x[i:j]

    def transpose_get (self, iroot, *inv):
        '''A single-rootspace slice of the SI vectors, transposed so that involved fragments
        are faster-moving
        
        Args:                   
            iroot: integer      
                Rootspace index 
            *inv: integers      
                Indices of nonspectator fragments
        
        Returns:
            sivec: ndarray of shape (nroots_si, nrows, ncols)
                SI vectors with the faster dimension iterating over states of fragments in
                inv and the slower dimension iterating over states of fragments not in inv 
        '''
        xvec = self.get_single_rootspace_x (iroot)
        ninv = [i for i in range (self.nfrags) if i not in inv]
        return transpose_sivec_make_fragments_slow (xvec, self.lroots[:,iroot], *ninv)[0]

    def transpose_put_(self, vec, iroot, *inv):
        ninv = [i for i in range (self.nfrags) if i not in inv]
        i, j = self.offs_lroots[iroot]
        self.ox[i:j] += transpose_sivec_with_slow_fragments (vec, self.lroots[:,iroot], *ninv)[0]
        return

    def _umat_linequiv_(self, ifrag, iroot, umat, ivec, *args):
        if ivec==0:
            self.x = umat_dot_1frag_(self.x, umat.conj ().T, self.lroots, ifrag, iroot)
        elif ivec==1:
            self.ox = umat_dot_1frag_(self.ox, umat, self.lroots, ifrag, iroot)
        else:
            raise RuntimeError ("Invalid ivec = {}; must be 0 or 1".format (ivec))

    def _init_crunch_put_profile (self):
        tzero = np.array ([0.0,0.0])
        self.crunch_put_profile = {'_crunch_1d_': tzero.copy (),
                                   '_crunch_2d_': tzero.copy (),
                                   '_crunch_1c_': tzero.copy (),
                                   '_crunch_1c1d_': tzero.copy (),
                                   '_crunch_1s_': tzero.copy (),
                                   '_crunch_1s1c_': tzero.copy (),
                                   '_crunch_2c_': tzero.copy (),
                                   ' get_vecs ': tzero.copy (),
                                   ' ovlpdot ': tzero.copy (),
                                   ' put_vecs ': tzero.copy (),
                                   ' op_data_transpose ': tzero.copy ()}

    def _ham_op (self, x):
        t0 = (logger.process_clock (), logger.perf_counter ())
        self.init_profiling ()
        self._init_crunch_put_profile ()
        self.x[:] = x.flat[:]
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
        self.log.debug1 (self.sprint_profile ())
        self.log.debug1 (str (self.crunch_put_profile))
        self.log.timer_debug1 ('HamS2OvlpOperators._ham_op', *t0)
        return self.ox.copy ()

    def _s2_op (self, x):
        t0 = (logger.process_clock (), logger.perf_counter ())
        self.init_profiling ()
        self._init_crunch_put_profile ()
        self.x[:] = x.flat[:]
        self.ox[:] = 0
        self._umat_linequiv_loop_(0) # U.conj () @ x
        for row in self.exc_1d: self._crunch_ox_env_(self._crunch_1d_, 1, *row)
        for row in self.exc_2d: self._crunch_ox_env_(self._crunch_2d_, 1, *row)
        for row in self.exc_1s: self._crunch_ox_env_(self._crunch_1s_, 1, *row)
        self._umat_linequiv_loop_(1) # U.T @ ox
        self.log.debug1 (self.sprint_profile ())
        self.log.debug1 (str (self.crunch_put_profile))
        self.log.timer_debug1 ('HamS2OvlpOperators._s2_op', *t0)
        return self.ox.copy ()

    def _crunch_ox_env_(self, _crunch_fn, opid, *row): 
        if self._fn_row_has_spin (_crunch_fn):
            inv = row[2:-1]     
        else:
            inv = row[2:]
        data = _crunch_fn (*row)
        op = data[opid]
        sinv = data[2]
        op = self.canonical_operator_order (op, sinv)
        key = tuple ((row[0], row[1])) + inv
        inv = list (set (inv))
        t0 = np.array ([logger.process_clock (), logger.perf_counter ()])
        opbralen = np.prod (self.lroots[inv,row[0]])
        opketlen = np.prod (self.lroots[inv,row[1]])
        op = op.reshape ((opbralen, opketlen), order='C')
        t1 = np.array ([logger.process_clock (), logger.perf_counter ()])
        self.crunch_put_profile[' op_data_transpose '] += (t1-t0)
        tab = self.nonuniq_exc[key]
        t0 = np.array ([logger.process_clock (), logger.perf_counter ()])
        bras, kets = self.nonuniq_exc[key].T
        self._put_ox_(bras, kets, op, inv, _conj=False)
        self._put_ox_(kets, bras, op.conj ().T, inv, _conj=True)
        t1 = np.array ([logger.process_clock (), logger.perf_counter ()])
        self.crunch_put_profile[_crunch_fn.__name__] += (t1-t0)
        return

    def _put_ox_(self, bras, kets, op, inv, _conj=False):
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        p0 = np.array ([logger.process_clock (), logger.perf_counter ()])
        uniq_kets, invrs_ket = np.unique (kets, return_inverse=True)
        uniq_bras, invrs_bra = np.unique (bras, return_inverse=True)
        ketvecs = [lib.dot (op, self.transpose_get (ket, *inv).T)
                   for ket in uniq_kets]
        p1 = np.array ([logger.process_clock (), logger.perf_counter ()])
        self.crunch_put_profile[' get_vecs '] += (p1-p0)
        bravecs = [0.0 for u in uniq_bras]
        for i in range (len (bras)):
            bra, ket, vec = bras[i], kets[i], ketvecs[invrs_ket[i]]
            vec = self.ox_ovlp_part (bra, ket, vec, inv, _conj=_conj)
            bravecs[invrs_bra[i]] += vec
        p2 = np.array ([logger.process_clock (), logger.perf_counter ()])
        self.crunch_put_profile[' ovlpdot '] += (p2-p1)
        for bra, vec in zip (uniq_bras, bravecs):
            self.transpose_put_(vec.ravel (), bra, *inv)
        p3 = np.array ([logger.process_clock (), logger.perf_counter ()])
        self.crunch_put_profile[' put_vecs '] += (p3-p2)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_p, self.dw_p = self.dt_p + dt, self.dw_p + dw

    def ox_ovlp_part (self, bra, ket, vec, inv, _conj=False):
        fac = self.spin_shuffle[bra] * self.spin_shuffle[ket]
        fac *= self.fermion_frag_shuffle (bra, inv)
        fac *= self.fermion_frag_shuffle (ket, inv)
        if (bra==ket): fac *= 0.5
        vec = fac * vec
        spec = np.ones (self.nfrags, dtype=bool)
        spec[inv] = False
        spec = np.where (spec)[0]
        for i in spec:
            lket = self.lroots[i,ket]
            vec = vec.reshape (-1,lket)
            o = self.ints[i].get_ovlp (bra, ket)
            if _conj: o = o.conj ()
            vec = lib.einsum ('pq,lq->pl', o, vec)
        return vec

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
        self.dt_p, self.dw_p = self.dt_p + dt, self.dw_p + dw

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

