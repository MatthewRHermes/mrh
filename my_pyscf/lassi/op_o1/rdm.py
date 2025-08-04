import time
import numpy as np
from scipy import linalg
from pyscf import lib
from pyscf.lib import logger, param
from mrh.my_pyscf.lassi.op_o1 import frag
from mrh.my_pyscf.lassi.op_o1 import stdm
from mrh.my_pyscf.lassi.op_o1.utilities import *

# C interface
import ctypes
from mrh.lib.helper import load_library
liblassi = load_library ('liblassi')
def c_arr (arr): return arr.ctypes.data_as(ctypes.c_void_p)
c_int = ctypes.c_int

class FragTDMInt (frag.FragTDMInt):
    __doc__ = frag.FragTDMInt.__doc__ + '''

    Transpose the transition density matrix arrays so that the lroots indices
    are fastest-moving, since RDMs require contracting over these indices
    '''

    def setmanip (self, x):
        x = np.ascontiguousarray (np.moveaxis (x, [0,1], [-2,-1]))
        return np.moveaxis (x, [-2,-1], [0,1])

class LRRDM (stdm.LSTDM):
    __doc__ = stdm.LSTDM.__doc__ + '''

    SUBCLASS: LASSI-root reduced density matrix

    `kernel` call returns reduced density matrices for LASSI eigenstates (or linear combinations of
    product states generally) without cacheing stdm12s array

    Additional args:
        si : ndarray of shape (nroots,nroots_si)
            Contains LASSI eigenvectors
    '''
    # TODO: at some point, if it ever becomes rate-limiting, make this multithread better
    # TODO: SO-LASSI o1 implementation: these density matrices can only be defined in the full
    # spinorbital basis

    def __init__(self, ints, nlas, lroots, si_bra, si_ket, mask_bra_space=None,
                 mask_ket_space=None, pt_order=None, do_pt_order=None, log=None,
                 max_memory=param.MAX_MEMORY, dtype=np.float64):
        stdm.LSTDM.__init__(self, ints, nlas, lroots,
                                mask_bra_space=mask_bra_space,
                                mask_ket_space=mask_ket_space,
                                pt_order=pt_order,
                                do_pt_order=do_pt_order,
                                log=log, max_memory=max_memory,
                                dtype=dtype)
        self.nroots_si = si_bra.shape[-1]
        self.si_bra = si_bra.copy ()
        self.si_ket = si_ket.copy ()
        self.hermi = np.amax (np.abs (si_bra-si_ket)) < 1e-8
        self._transpose = False
        self._umat_linequiv_loop_(self.si_bra)
        self._umat_linequiv_loop_(self.si_ket)
        self.si_bra = np.asfortranarray (self.si_bra)
        self.si_ket = np.asfortranarray (self.si_ket)
        assert (self.si_bra.shape == self.si_ket.shape)
        self._si_c_nrow = c_int (self.si_bra.shape[0])
        self._si_c_ncol = c_int (self.si_bra.shape[1])
        self.d1buf = self.d1 = np.empty ((self.nroots_si,self.d1.size), dtype=self.d1.dtype)
        self.d2buf = self.d2 = np.empty ((self.nroots_si,self.d2.size), dtype=self.d2.dtype)
        self._d1buf_c = c_arr (self.d1buf)
        self._d2buf_c = c_arr (self.d2buf)

    def _add_transpose_(self):
        if self.hermi:
            self.rdm1s += self.rdm1s.conj ().transpose (0,1,3,2)
            self.rdm2s += self.rdm2s.conj ().transpose (0,1,3,2,5,4)

    def _umat_linequiv_(self, ifrag, iroot, umat, *args):
        si = args[0]
        return umat_dot_1frag_(si, umat.conj ().T, self.lroots, ifrag, iroot, axis=0)

    def kernel (self):
        ''' Main driver method of class.

        Returns:
            rdm1s : ndarray of shape (nroots_si,2,ncas,ncas)
                Spin-separated 1-body reduced density matrices of LASSI states
            rdm2s : ndarray of shape (nroots_si,4,ncas,ncas,ncas,ncas)
                Spin-separated 2-body reduced density matrices of LASSI states
            t0 : tuple of length 2
                timestamp of entry into this function, for profiling by caller
        '''
        t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
        self.init_profiling ()
        self.rdm1s = np.zeros ([self.nroots_si,2] + [self.norb,]*2, dtype=self.dtype)
        self.rdm2s = np.zeros ([self.nroots_si,4] + [self.norb,]*4, dtype=self.dtype)
        self._rdm1s_c = c_arr (self.rdm1s)
        self._rdm1s_c_ncol = c_int (2*(self.norb**2))
        self._rdm2s_c = c_arr (self.rdm2s)
        self._rdm2s_c_ncol = c_int (4*(self.norb**4))
        self._crunch_all_()
        return self.rdm1s, self.rdm2s, t0

    def init_profiling (self):
        self.dt_1d, self.dw_1d = 0.0, 0.0
        self.dt_2d, self.dw_2d = 0.0, 0.0
        self.dt_1c, self.dw_1c = 0.0, 0.0
        self.dt_1c1d, self.dw_1c1d = 0.0, 0.0
        self.dt_1s, self.dw_1s = 0.0, 0.0
        self.dt_1s1c, self.dw_1s1c = 0.0, 0.0
        self.dt_2c, self.dw_2c = 0.0, 0.0
        self.dt_o, self.dw_o = 0.0, 0.0
        self.dt_u, self.dw_u = 0.0, 0.0 
        self.dt_p, self.dw_p = 0.0, 0.0 
        self.dt_i, self.dw_i = 0.0, 0.0 

    def sprint_profile (self):
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
        profile += '\n' + fmt_str.format ('put', self.dt_p, self.dw_p)
        profile += '\n' + fmt_str.format ('idx', self.dt_i, self.dw_i)
        return profile

    def get_single_rootspace_sivec (self, iroot, bra=False):
        '''A single-rootspace slice of the SI vectors.

        Args:
            iroot: integer
                Rootspace index

        Returns:
            sivec: col-major ndarray of shape (np.prod (lroots[:,iroot], nroots_si)
                SI vectors
        '''
        i, j = self.offs_lroots[iroot]
        if self._transpose: bra = not bra
        si = self.si_bra if bra else self.si_ket
        return si[i:j,:]

    def get_frag_transposed_sivec (self, iroot, *inv, bra=False):
        '''A single-rootspace slice of the SI vectors, transposed so that involved fragments
        are slower-moving

        Args:
            iroot: integer
                Rootspace index
            *inv: integers 
                Indices of nonspectator fragments

        Returns:
            sivec: ndarray of shape (nroots_si, nrows, ncols)
                SI vectors with the faster dimension iterating over states of fragments not in
                inv and the slower dimension iterating over states of fragments in inv 
        '''
        sivec = self.get_single_rootspace_sivec (iroot, bra=bra)
        return transpose_sivec_make_fragments_slow (sivec, self.lroots[:,iroot], *inv)

    def get_fdm (self, rbra, rket, *inv, keyorder=None, _braket_table=None):
        '''Get the n-fragment density matrices for the fragments identified by inv in the bra and
        spaces given by rbra and rket, summing over nonunique excitations

        Args: 
            rbra: integer
                Index of bra rootspace for which to prepare the current cache.
            rket: integer
                Index of ket rootspace for which to prepare the current cache.
            *inv: integers 
                Indices of nonspectator fragments

        Kwargs:
            keyorder: list of integers
                The same fragments as inv in a different order, in case the key in self.nonuniq_exc
                uses a different order than desired in the output
            _braket_table: ndarray of shape (*,2)
                Overrides the lookup of self.nonuniq_exc if provided.

        Returns:
            fdm : ndarray of shape (nroots_si, ..., lroots[inv[1],rbra], lroots[inv[1],rket],
                                    lroots[inv[0],rbra], lroots[inv[0],rket])
                len(inv)-fragment reduced density matrix
        '''
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        if keyorder is None: keyorder = inv
        key = tuple ((rbra,rket)) + tuple (keyorder)
        if _braket_table is None:
            braket_table = self.nonuniq_exc[key]
        else:
            braket_table = _braket_table
        invset = set ()
        inv = [i for i in inv if not (i in invset or invset.add (i))]
        # must eliminate duplicates, but must also preserve order
        fdm = 0
        for rbra1, rket1 in braket_table:
            fdm += self.get_fdm_1space (rbra1, rket1, *inv)
        fdm = np.ascontiguousarray (fdm)
        newshape = [self.nroots_si,] + list (self.lroots[inv,rbra][::-1]) + list (self.lroots[inv,rket][::-1])
        fdm = fdm.reshape (newshape)
        axesorder = [0,] + sum ([[i+1, i+1+len(inv)] for i in range (len(inv))], [])
        fdm = fdm.transpose (*axesorder)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_o, self.dw_o = self.dt_o + dt, self.dw_o + dw
        return np.ascontiguousarray (fdm)

    def get_fdm_1space (self, rbra, rket, *inv):
        '''Get the n-fragment density matrices for the fragments identified by inv in the bra and
        spaces given by rbra and rket. Not necessarily meaningful by itself because basis states
        can appear in multiple rootspaces, so there are multiple (bra,ket) tuples which can
        correspond to the same interaction.

        Args:
            rbra: integer
                Index of a rootspace
            rket: integer
                Index of a rootspace
            *inv: integers
                Indices of nonspectator fragments.

        Returns:
            fdm : ndarray of shape (nroots_si, prod (lroots[inv,rbra]), prod (lroots[inv,rket]))
                len(inv)-fragment reduced density matrix
        '''
        # TODO: get rid of the outer product of overlap matrices here? My first attempt made it
        # slower though, I guess because of the line filtering out zero-overlap elements.
        sibra = self.get_frag_transposed_sivec (rbra, *inv, bra=True)
        siket = self.get_frag_transposed_sivec (rket, *inv, bra=False)
        inv = list (set (inv))
        fac = self.spin_shuffle[rbra] * self.spin_shuffle[rket]
        fac *= self.fermion_frag_shuffle (rbra, inv)
        fac *= self.fermion_frag_shuffle (rket, inv)
        spec = np.ones (self.nfrags, dtype=bool)
        for i in inv: spec[i] = False
        spec = np.where (spec)[0]
        specints = [self.ints[i] for i in spec]
        o = fac * np.ones ((1,1), dtype=self.dtype)
        for i in specints:
            o = np.multiply.outer (i.get_ovlp (rbra, rket), o).transpose (0,2,1,3)
            o = o.reshape (o.shape[0]*o.shape[1], o.shape[2]*o.shape[3])
        idx = np.abs(o) > 1e-8
        if self.ltri and (rbra==rket): # not bra==ket b/c _loop_lroots_ isn't tril
            o[np.diag_indices_from (o)] *= 0.5
            idx[np.triu_indices_from (idx, k=1)] = False
        o = o[idx]
        b, k = np.where (idx)
        # Numpy pads array dimension to the left
        sibra = sibra[:,:,b].conj () * o
        siket = siket[:,:,k]
        fdm = np.stack ([np.dot (b, k.T) for b, k in zip (sibra, siket)], axis=0)
        return fdm
                
    def _crunch_1d_(self, bra, ket, i):
        '''Compute a single-fragment density fluctuation, for both the 1- and 2-RDMs.'''
        d_rII = self.get_fdm (bra, ket, i) # time-profiled by itself
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        inti = self.ints[i]
        d1 = self._get_D1_(bra, ket)
        d2 = self._get_D2_(bra, ket)
        p, q = self.get_range (i)
        d1[:,:,p:q,p:q] = np.tensordot (d_rII, inti.get_dm1 (bra, ket), axes=2)
        d2[:,:,p:q,p:q,p:q,p:q] = np.tensordot (d_rII, inti.get_dm2 (bra, ket), axes=2)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_1d, self.dw_1d = self.dt_1d + dt, self.dw_1d + dw
        self._put_D1_()
        self._put_D2_()

    def _crunch_2d_(self, bra, ket, i, j):
        '''Compute a two-fragment density fluctuation.'''
        d_ = self.get_fdm (bra, ket, i, j) # time-profiled by itself
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        d2 = self._get_D2_(bra, ket)
        inti, intj = self.ints[i], self.ints[j]
        p, q = self.get_range (i)
        r, s = self.get_range (j)
        # d_rJJII
        d_ = np.tensordot (d_, inti.get_dm1 (bra, ket), axes=2) # _rJJsii
        d_ = np.tensordot (d_, intj.get_dm1 (bra, ket), axes=((1,2),(0,1))) # _rsiisjj
        d_ = d_.transpose (0,1,4,2,3,5,6).reshape (self.nroots_si, 4, q-p, q-p, s-r, s-r) # _rsiijj
        d2[:,:,p:q,p:q,r:s,r:s] = d_
        d2[:,(0,3),r:s,r:s,p:q,p:q] = d_[:,(0,3),...].transpose (0,1,4,5,2,3)
        d2[:,(1,2),r:s,r:s,p:q,p:q] = d_[:,(2,1),...].transpose (0,1,4,5,2,3)
        d2[:,(0,3),p:q,r:s,r:s,p:q] = -d_[:,(0,3),...].transpose (0,1,2,5,4,3)
        d2[:,(0,3),r:s,p:q,p:q,r:s] = -d_[:,(0,3),...].transpose (0,1,4,3,2,5)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_2d, self.dw_2d = self.dt_2d + dt, self.dw_2d + dw
        self._put_D2_()

    def _crunch_1c_(self, bra, ket, i, j, s1):
        '''Compute the reduced density matrix elements of a single electron hop; i.e.,
    
        <bra|j'(s1)i(s1)|ket>
    
        i.e.,
    
        j ---s1---> i
    
        and conjugate transpose
        '''
        d_rJJII = self.get_fdm (bra, ket, i, j) # time-profiled by itself
        t0, w0 = logger.process_clock (), logger.perf_counter () 
        d1 = self._get_D1_(bra, ket)
        d2 = self._get_D2_(bra, ket)
        inti, intj = self.ints[i], self.ints[j]
        p, q = self.get_range (i)
        r, s = self.get_range (j)
        fac = 1                 
        nelec_f_bra = self.nelec_rf[bra]
        nelec_f_ket = self.nelec_rf[ket]
        fac *= fermion_des_shuffle (nelec_f_bra, (i, j), i)
        fac *= fermion_des_shuffle (nelec_f_ket, (i, j), j)
        d_riJJ = np.tensordot (d_rJJII, inti.get_p (bra, ket, s1), axes=2).transpose (0,3,1,2)
        d1[:,s1,p:q,r:s] = fac * np.tensordot (d_riJJ, intj.get_h (bra, ket, s1), axes=2)
        s12l = s1 * 2   # aa: 0 OR ba: 2
        s12h = s12l + 1 # ab: 1 OR bb: 3 
        s21l = s1       # aa: 0 OR ab: 1
        s21h = s21l + 2 # ba: 2 OR bb: 3
        s1s1 = s1 * 3   # aa: 0 OR bb: 3
        def _crunch_1c_tdm2 (d2_rsijkk, i0, i1, j0, j1, k0, k1):
            d2[:,(s12l,s12h), i0:i1, j0:j1, k0:k1, k0:k1] = d2_rsijkk
            d2[:,(s21l,s21h), k0:k1, k0:k1, i0:i1, j0:j1] = d2_rsijkk.transpose (0,1,4,5,2,3)
            d2[:,s1s1, i0:i1, k0:k1, k0:k1, j0:j1] = -d2_rsijkk[:,s1,...].transpose (0,1,4,3,2)
            d2[:,s1s1, k0:k1, j0:j1, i0:i1, k0:k1] = -d2_rsijkk[:,s1,...].transpose (0,3,2,1,4)
        # phh (transpose to bring spin to outside and then from Dirac order to Mulliken order)
        d_ = np.tensordot (d_riJJ, intj.get_phh (bra, ket, s1), axes=2) # _risjjj'
        d_ = fac * d_.transpose (0,2,1,5,3,4) # _rsij'jj
        _crunch_1c_tdm2 (d_, p, q, r, s, r, s)
        # pph (transpose from Dirac order to Mulliken order)
        d_ = np.tensordot (d_rJJII, inti.get_pph (bra, ket, s1), axes=2) # _rJJsi'ii
        d_ = d_.transpose (0,3,4,5,6,1,2) # _rsi'iiJJ
        d_ = np.tensordot (d_, intj.get_h (bra, ket, s1), axes=2) # _rsi'iij
        d_ = fac * d_.transpose (0,1,2,5,3,4) # _rsi'jii
        _crunch_1c_tdm2 (d_, p, q, r, s, p, q)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_1c, self.dw_1c = self.dt_1c + dt, self.dw_1c + dw
        self._put_D1_()
        self._put_D2_()

    def _crunch_1c1d_(self, bra, ket, i, j, k, s1):
        '''Compute the reduced density matrix elements of a coupled electron-hop and
        density fluctuation.'''
        d_ = self.get_fdm (bra, ket, i, j, k) # time-profiled by itself
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        d2 = self._get_D2_(bra, ket)
        inti, intj, intk = self.ints[i], self.ints[j], self.ints[k]
        p, q = self.get_range (i)
        r, s = self.get_range (j)
        t, u = self.get_range (k)
        fac = 1
        nelec_f_bra = self.nelec_rf[bra]
        nelec_f_ket = self.nelec_rf[ket]
        fac *= fermion_des_shuffle (nelec_f_bra, (i, j, k), i)
        fac *= fermion_des_shuffle (nelec_f_ket, (i, j, k), j)
        s12l = s1 * 2   # aa: 0 OR ba: 2
        s12h = s12l + 1 # ab: 1 OR bb: 3 
        s21l = s1       # aa: 0 OR ab: 1
        s21h = s21l + 2 # ba: 2 OR bb: 3
        s1s1 = s1 * 3   # aa: 0 OR bb: 3
        def _crunch_1c_tdm2 (d_rsijkk, i0, i1, j0, j1, k0, k1):
            d2[:,(s12l,s12h), i0:i1, j0:j1, k0:k1, k0:k1] = d_rsijkk
            d2[:,(s21l,s21h), k0:k1, k0:k1, i0:i1, j0:j1] = d_rsijkk.transpose (0,1,4,5,2,3)
            d2[:,s1s1, i0:i1, k0:k1, k0:k1, j0:j1] = -d_rsijkk[:,s1,...].transpose (0,1,4,3,2)
            d2[:,s1s1, k0:k1, j0:j1, i0:i1, k0:k1] = -d_rsijkk[:,s1,...].transpose (0,3,2,1,4)
        # d_rKKJJII
        d_ = np.tensordot (d_, inti.get_p (bra, ket, s1), axes=2) # d_rKKJJi
        d_ = np.tensordot (d_, intj.get_h (bra, ket, s1), axes=((-3,-2),(0,1))) # d_rKKij
        d_ = np.tensordot (d_, intk.get_dm1 (bra, ket), axes=((-4,-3),(0,1))) # d_rijskk
        d_ = fac * d_.transpose (0,3,1,2,4,5) # d_rsijkk
        _crunch_1c_tdm2 (d_, p, q, r, s, t, u)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_1c1d, self.dw_1c1d = self.dt_1c1d + dt, self.dw_1c1d + dw
        self._put_D2_()

    def _crunch_1s_(self, bra, ket, i, j):
        '''Compute the reduced density matrix elements of a spin unit hop; i.e.,

        <bra|i'(a)j'(b)i(b)j(a)|ket>

        i.e.,

        j ---a---> i
        i ---b---> j

        and conjugate transpose
        '''
        d_ = self.get_fdm (bra, ket, i, j) # time-profiled by itself
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        inti, intj = self.ints[i], self.ints[j]
        d2 = self._get_D2_(bra, ket) # aa, ab, ba, bb -> 0, 1, 2, 3
        p, q = self.get_range (i)
        r, s = self.get_range (j)
        fac = -1
        d_ = np.tensordot (d_, inti.get_sp (bra, ket), axes=2).transpose (0,3,4,1,2)
        d_ = fac * np.tensordot (d_, intj.get_sm (bra, ket), axes=2)
        d2[:,1,p:q,r:s,r:s,p:q] = d_.transpose (0,1,4,3,2)
        d2[:,2,r:s,p:q,p:q,r:s] = d_.transpose (0,3,2,1,4)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_1s, self.dw_1s = self.dt_1s + dt, self.dw_1s + dw
        self._put_D2_()

    def _crunch_1s1c_(self, bra, ket, i, j, k, s1):
        '''Compute the reduced density matrix elements of a spin-charge unit hop; i.e.,

        <bra|i'(a)k'(b)j(b)k(a)|ket>

        i.e.,

        k ---a---> i
        j ---b---> k

        and conjugate transpose
        '''
        d_ = self.get_fdm (bra, ket, i, j, k) # time-profiled by itself
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        s2 = 1 - s1
        d2 = self._get_D2_(bra, ket) # aa, ab, ba, bb -> 0, 1, 2, 3
        p, q = self.get_range (i)
        r, s = self.get_range (j)
        t, u = self.get_range (k)
        nelec_f_bra = self.nelec_rf[bra]
        nelec_f_ket = self.nelec_rf[ket]
        fac = -1 # a'bb'a -> a'ab'b sign
        fac *= fermion_des_shuffle (nelec_f_bra, (i, j, k), i)
        fac *= fermion_des_shuffle (nelec_f_ket, (i, j, k), j)
        d_ = np.tensordot (d_, self.ints[i].get_p (bra, ket, s1), axes=2) # _rKKJJi
        d_ = np.tensordot (d_, self.ints[j].get_h (bra, ket, s2), axes=((3,4),(0,1))) # _rKKij
        if s1 == 0:
            d_ = np.tensordot (d_, self.ints[k].get_sm (bra, ket), axes=((1,2),(0,1))) # _rijk'k
        else:
            d_ = np.tensordot (d_, self.ints[k].get_sp (bra, ket), axes=((1,2),(0,1))) # _rijk'k
        d_ = fac * d_.transpose (0,1,4,3,2) # r_ikk'j (a'bb'a -> a'ab'b transpose)
        d2[:,1+s1,p:q,t:u,t:u,r:s] = d_ #rikkj
        d2[:,2-s1,t:u,r:s,p:q,t:u] = d_.transpose (0,3,4,1,2)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_1s1c, self.dw_1s1c = self.dt_1s1c + dt, self.dw_1s1c + dw
        self._put_D2_()

    def _crunch_2c_(self, bra, ket, i, j, k, l, s2lt):
        '''Compute the reduced density matrix elements of a two-electron hop; i.e.,

        <bra|i'(s1)k'(s2)l(s2)j(s1)|ket>

        i.e.,

        j ---s1---> i
        l ---s2---> k

        with

        s2lt = 0, 1, 2
        s1   = a, a, b
        s2   = a, b, b

        and conjugate transpose

        Note that this includes i=k and/or j=l cases, but no other coincident fragment indices. Any
        other coincident fragment index (that is, any coincident index between the bra and the ket)
        turns this into one of the other interactions implemented in the above _crunch_ functions:
        s1 = s2  AND SORT (ik) = SORT (jl)                 : _crunch_1d_ and _crunch_2d_
        s1 = s2  AND (i = j XOR i = l XOR j = k XOR k = l) : _crunch_1c_ and _crunch_1c1d_
        s1 != s2 AND (i = l AND j = k)                     : _crunch_1s_
        s1 != s2 AND (i = l XOR j = k)                     : _crunch_1s1c_
        '''
        d_ = self.get_fdm (bra, ket, i, k, l, j, keyorder=[i,j,k,l]) # time-profiled by itself
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        # s2lt: 0, 1, 2 -> aa, ab, bb
        # s2: 0, 1, 2, 3 -> aa, ab, ba, bb
        s2  = (0, 1, 3)[s2lt] # aa, ab, bb
        s2T = (0, 2, 3)[s2lt] # aa, ba, bb -> when you populate the e1 <-> e2 permutation
        s11 = s2 // 2
        s12 = s2 % 2
        nelec_f_bra = self.nelec_rf[bra]
        nelec_f_ket = self.nelec_rf[ket]
        d2 = self._get_D2_(bra, ket)
        fac = 1
        if i == k: # d_r...II
            d_ = np.tensordot (d_, self.ints[i].get_pp (bra, ket, s2lt), axes=2) # _r...ik
        else: # d_r...KKII
            d_ = np.tensordot (d_, self.ints[i].get_p (bra, ket, s11), axes=2) # _r...KKi
            d_ = np.tensordot (d_, self.ints[k].get_p (bra, ket, s12),
                               axes=((-3,-2),(0,1))) # _r...ik
            fac *= (1,-1)[int (i>k)]
            fac *= fermion_des_shuffle (nelec_f_bra, (i, j, k, l), i)
            fac *= fermion_des_shuffle (nelec_f_bra, (i, j, k, l), k)
        axesorder = [0,d_.ndim-2,d_.ndim-1] + list (range (1, d_.ndim-2))
        d_ = d_.transpose (*axesorder) # _rik...
        if j == l: # d_rikJJ
            d_ = np.tensordot (d_, self.ints[j].get_hh (bra, ket, s2lt), axes=2) # _riklj
        else: # d_rikJJLL
            d_ = np.tensordot (d_, self.ints[l].get_h (bra, ket, s12), axes=2) # _rikJJl
            d_ = np.tensordot (d_, self.ints[j].get_h (bra, ket, s11),
                               axes=((-3,-2),(0,1))) # r_iklj
            fac *= (1,-1)[int (j>l)]
            fac *= fermion_des_shuffle (nelec_f_ket, (i, j, k, l), j)
            fac *= fermion_des_shuffle (nelec_f_ket, (i, j, k, l), l)
        d_ = fac * d_.transpose (0,1,4,2,3) # _rijkl (Dirac -> Mulliken transp_
        p, q = self.get_range (i)
        r, s = self.get_range (j)
        t, u = self.get_range (k)
        v, w = self.get_range (l)
        d2[:,s2, p:q,r:s,t:u,v:w] = d_
        d2[:,s2T,t:u,v:w,p:q,r:s] = d_.transpose (0,3,4,1,2)
        if s2 == s2T: # same-spin only: exchange happens
            d2[:,s2,p:q,v:w,t:u,r:s] = -d_.transpose (0,1,4,3,2)
            d2[:,s2,t:u,r:s,p:q,v:w] = -d_.transpose (0,3,2,1,4)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_2c, self.dw_2c = self.dt_2c + dt, self.dw_2c + dw
        self._put_D2_()

    def _put_D1_(self):
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        if self._transpose: self.d1[:] = self.d1.transpose (0,1,3,2).conj ()
        fn = self._put_SD1_c_fn
        fn (self._rdm1s_c, self._d1buf_c,
            self._si_c_ncol, self._norb_c, self._nsrc_c,
            self._dblk_idx, self._sblk_idx, self._lblk, self._nblk)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_p, self.dw_p = self.dt_p + dt, self.dw_p + dw

    def _put_D2_(self):
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        if self._transpose: self.d2[:] = self.d2.transpose (0,1,3,2,5,4).conj ()
        fn = self._put_SD2_c_fn
        fn (self._rdm2s_c, self._d2buf_c,
            self._si_c_ncol, self._norb_c, self._nsrc_c, self._pdest,
            self._dblk_idx, self._sblk_idx, self._lblk, self._nblk)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_p, self.dw_p = self.dt_p + dt, self.dw_p + dw

    def _crunch_env_(self, _crunch_fn, *row):
        if self._fn_row_has_spin (_crunch_fn):
            inv = row[2:-1]
        else:
            inv = row[2:]
        env_kwargs = self._orbrange_env_kwargs (inv)
        with lib.temporary_env (self, **env_kwargs):
            _crunch_fn (*row)
        if not self.hermi:
            env_kwargs['_transpose'] = True
            with lib.temporary_env (self, **env_kwargs):
                _crunch_fn (*row)

    def _orbrange_env_kwargs (self, inv):
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        fragidx = np.zeros (self.nfrags, dtype=bool)
        _orbidx = np.zeros (self.norb, dtype=bool)
        for frag in inv:
            fragidx[frag] = True
            p, q = self.get_range (frag)
            _orbidx[p:q] = True
        nlas = np.array (self.nlas)
        nlas[~fragidx] = 0
        norb = sum (nlas)
        d1 = self.d1
        if len (inv) < 3: # Otherwise this won't be touched anyway
            d1_shape = [self.nroots_si,] + [2,] + [norb,]*2
            d1_size = np.prod (d1_shape)
            d1 = self.d1.ravel ()[:d1_size].reshape (d1_shape)
        d2_shape = [self.nroots_si,] + [4,] + [norb,]*4
        d2_size = np.prod (d2_shape)
        d2 = self.d2.ravel ()[:d2_size].reshape (d2_shape)
        env_kwargs = {'nlas': nlas, 'd1': d1, 'd2': d2, '_orbidx': _orbidx}
        env_kwargs.update (self._orbrange_env_kwargs_orbidx (_orbidx))
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_i, self.dw_i = self.dt_i + dt, self.dw_i + dw
        return env_kwargs

def get_fdm1_maker (las, ci, nelec_frs, si, **kwargs):
    ''' Get a function that can build the 1-fragment reduced density matrix
    in a single rootspace. For unittesting purposes (make_sdm1 in sitools does the same thing)

    Args:
        las : instance of :class:`LASCINoSymm`
        ci : list of list of ndarrays 
            Contains all CI vectors
        nelec_frs : ndarray of shape (nfrags,nroots,2)
            Number of electrons of each spin in each rootspace in each
            fragment
        si : ndarray of shape (nroots,nroots_si)
            Contains LASSI eigenvectors
        
    Returns:
        make_fdm1 : callable
            Args:
                i : integer
                    Rootspace index
                j : integer
                    Fragment index
            Returns:
                fdm : ndarray of shape (si.shape[1], lroots[j,i], lroots[j,i])
                    1-fragment reduced density matrix
    ''' 
    verbose = kwargs.get ('verbose', las.verbose)
    log = logger.new_logger (las, verbose)
    nlas = las.ncas_sub
    ncas = las.ncas 
    nroots_si = si.shape[-1]
    max_memory = getattr (las, 'max_memory', las.mol.max_memory)
    dtype = ci[0][0].dtype 
        
    # First pass: single-fragment intermediates
    ints, lroots = frag.make_ints (las, ci, nelec_frs, nlas=nlas)
    nstates = np.sum (np.prod (lroots, axis=0))
        
    # Second pass: upper-triangle
    outerprod = LRRDM (ints, nlas, lroots, si, si, dtype=dtype,
                          max_memory=max_memory, log=log)

    # Spoof nonuniq_exc to avoid summing together things that need to be separate
    outerprod.ltri = False
    def make_fdm1 (iroot, ifrag):
        fdm = outerprod.get_fdm_1space (iroot, iroot, ifrag)
        if iroot in ints[ifrag].umat_root:
            umat = ints[ifrag].umat_root[iroot]
            fdm = lib.einsum ('rij,ik,jl->rkl',fdm,umat,umat)
        return fdm
    return make_fdm1

def roots_trans_rdm12s (las, ci, nelec_frs, si_bra, si_ket, **kwargs):
    ''' Build spin-separated LASSI 1- and 2-body transition reduced density matrices

    Args:
        las : instance of :class:`LASCINoSymm`
        ci : list of list of ndarrays
            Contains all CI vectors
        nelec_frs : ndarray of shape (nfrags,nroots,2)
            Number of electrons of each spin in each rootspace in each
            fragment
        si_bra : ndarray of shape (nroots,nroots_si)
            Contains LASSI eigenvectors for the bra
        si_ket : ndarray of shape (nroots,nroots_si)
            Contains LASSI eigenvectors for the ket

    Returns:
        rdm1s : ndarray of shape (nroots_si,2,ncas,ncas)
            Spin-separated 1-body reduced density matrices of LASSI states
        rdm2s : ndarray of shape (nroots_si,2,ncas,ncas,2,ncas,ncas)
            Spin-separated 2-body reduced density matrices of LASSI states
    '''
    verbose = kwargs.get ('verbose', las.verbose)
    log = lib.logger.new_logger (las, verbose)
    nlas = las.ncas_sub
    ncas = las.ncas
    pt_order = kwargs.get ('pt_order', None)
    do_pt_order = kwargs.get ('do_pt_order', None)
    assert (si_bra.dtype == si_ket.dtype)
    assert (si_bra.shape == si_ket.shape)
    nroots_si = si_ket.shape[-1]
    max_memory = getattr (las, 'max_memory', las.mol.max_memory)
    dtype = si_ket.dtype
    nfrags, nroots = nelec_frs.shape[:2]
    #if np.iscomplexobj (si):
    #    raise RuntimeError ("Known bug here with complex SI vectors")

    # Handle possible SOC
    nelec_rs = [tuple (x) for x in nelec_frs.sum (0)]
    spin_pure = len (set (nelec_rs)) == 1
    if not spin_pure: # Engage the ``spinless mapping''
        ci = ci_map2spinless (ci, nlas, nelec_frs)
        ix = spin_shuffle_idx (nlas)
        nlas = [2*x for x in nlas]
        spin_shuffle_fac = [fermion_spin_shuffle (nelec_frs[:,i,0], nelec_frs[:,i,1])
                            for i in range (nroots)]
        nelec_frs[:,:,0] += nelec_frs[:,:,1]
        nelec_frs[:,:,1] = 0
        ncas = ncas * 2

    # First pass: single-fragment intermediates
    ints, lroots = frag.make_ints (las, ci, nelec_frs, nlas=nlas,
                                                  _FragTDMInt_class=FragTDMInt,
                                                  pt_order=pt_order,
                                                  do_pt_order=do_pt_order)
    nstates = np.sum (np.prod (lroots, axis=0))
    
    # Memory check
    current_memory = lib.current_memory ()[0]
    required_memory = dtype.itemsize*nroots_si*(2*(ncas**2)+4*(ncas**4))/1e6
    if current_memory + required_memory > max_memory:
        raise MemoryError ("current: {}; required: {}; max: {}".format (
            current_memory, required_memory, max_memory))

    # Second pass: upper-triangle
    t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
    outerprod = LRRDM (ints, nlas, lroots, si_bra, si_ket,
                       pt_order=pt_order, do_pt_order=do_pt_order,
                       dtype=dtype, max_memory=max_memory, log=log)

    if not spin_pure:
        outerprod.spin_shuffle = spin_shuffle_fac
    lib.logger.timer (las, 'LASSI root RDM12s second intermediate indexing setup', *t0)
    rdm1s, rdm2s, t0 = outerprod.kernel ()
    lib.logger.timer (las, 'LASSI root RDM12s second intermediate crunching', *t0)
    if las.verbose >= lib.logger.TIMER_LEVEL:
        lib.logger.info (las, 'LASSI root RDM12s crunching profile:\n%s', outerprod.sprint_profile ())

    # Clean up the ``spinless mapping''
    if not spin_pure:
        # TODO: 2e- SOC
        kx = [True,]*2
        jx = [True,]*nroots_si
        rdm1s = rdm1s[np.ix_(jx,kx,ix,ix)]
        rdm2s = rdm2s[np.ix_(jx,kx*2,ix,ix,ix,ix)]
        n = ncas = ncas // 2
        rdm2s_ = np.zeros ((nroots_si, 2, 2, n, n, n, n), dtype=rdm2s.dtype)
        rdm2s_[:,0,0,:,:,:,:] = rdm2s[:,0,:n,:n,:n,:n]
        rdm2s_[:,0,1,:,:,:,:] = rdm2s[:,0,:n,:n,n:,n:]
        rdm2s_[:,1,0,:,:,:,:] = rdm2s[:,0,n:,n:,:n,:n]
        rdm2s_[:,1,1,:,:,:,:] = rdm2s[:,0,n:,n:,n:,n:]
        rdm2s = rdm2s_

    # Put rdm1s in PySCF convention: [p,q] -> q'p
    if spin_pure: rdm1s = rdm1s.transpose (0,1,3,2)
    else: rdm1s = rdm1s[:,0].transpose (0,2,1)
    rdm2s = rdm2s.reshape (nroots_si, 2, 2, ncas, ncas, ncas, ncas).transpose (0,1,3,4,2,5,6).conj ()

    return rdm1s, rdm2s


def roots_make_rdm12s (las, ci, nelec_frs, si, **kwargs):
    ''' Build spin-separated LASSI 1- and 2-body reduced density matrices

    Args:
        las : instance of :class:`LASCINoSymm`
        ci : list of list of ndarrays
            Contains all CI vectors
        nelec_frs : ndarray of shape (nfrags,nroots,2)
            Number of electrons of each spin in each rootspace in each
            fragment
        si : ndarray of shape (nroots,nroots_si)
            Contains LASSI eigenvectors

    Returns:
        rdm1s : ndarray of shape (nroots_si,2,ncas,ncas)
            Spin-separated 1-body reduced density matrices of LASSI states
        rdm2s : ndarray of shape (nroots_si,2,ncas,ncas,2,ncas,ncas)
            Spin-separated 2-body reduced density matrices of LASSI states
    '''
    return roots_trans_rdm12s (las, ci, nelec_frs, si, si, **kwargs)        

