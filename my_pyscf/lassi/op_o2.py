import time
import numpy as np
from scipy import linalg
from pyscf import lib
from pyscf.lib import logger
from mrh.my_pyscf.lassi import op_o1

# C interface
import ctypes
from mrh.lib.helper import load_library
liblassi = load_library ('liblassi')
def c_arr (arr): return arr.ctypes.data_as(ctypes.c_void_p)
c_int = ctypes.c_int

make_stdm12s = op_o1.make_stdm12s
ham = op_o1.ham
contract_ham_ci = op_o1.contract_ham_ci
fermion_frag_shuffle = op_o1.fermion_frag_shuffle
fermion_des_shuffle = op_o1.fermion_des_shuffle

class LRRDMint (op_o1.LRRDMint):
    __doc__ = op_o1.LRRDMint.__doc__ + '''

    op_o2 reimplementation: get rid of outer products! This will take a while...
    '''

    def __init__(self, ints, nlas, hopping_index, lroots, si, mask_bra_space=None,
                 mask_ket_space=None, log=None, max_memory=2000, dtype=np.float64):
        op_o1.LSTDMint2.__init__(self, ints, nlas, hopping_index, lroots,
                                 mask_bra_space=mask_bra_space,
                                 mask_ket_space=mask_ket_space,
                                 log=log, max_memory=max_memory,
                                 dtype=dtype)
        self.nroots_si = si.shape[-1]
        self.si = si.copy ()
        self._umat_linequiv_loop_(self.si)
        self.si = np.asfortranarray (self.si)
        self._si_c = c_arr (self.si)
        self._si_c_nrow = c_int (self.si.shape[0])
        self._si_c_ncol = c_int (self.si.shape[1])
        self.d1buf = self.d1 = np.empty ((self.nroots_si,self.d1.size), dtype=self.d1.dtype)
        self.d2buf = self.d2 = np.empty ((self.nroots_si,self.d2.size), dtype=self.d2.dtype)
        self._d1buf_c = c_arr (self.d1buf)
        self._d2buf_c = c_arr (self.d2buf)

    def get_single_rootspace_sivec (self, iroot):
        '''A single-rootspace slice of the SI vectors, reshaped to expose the lroots.

        Args:
            iroot: integer
                Rootspace index

        Returns:
            sivec: col-major ndarray of shape (lroots[0,iroot], lroots[1,iroot], ...,
                                               nroots_si)
                SI vectors
        '''
        i, j = self.offs_lroots[iroot]
        vecshape = list (self.lroots[:,iroot]) + [self.nroots_si,]
        return self.si[i:j,:].reshape (vecshape, order='F')

    _lowertri = True

    def get_frag_transposed_sivec (self, iroot, *inv):
        '''A single-rootspace slice of the SI vectors, transposed so that involved fragments
        are slower-moving

        Args:
            iroot: integer
                Rootspace index
            *inv: integers 
                Indices of nonspectator fragments

        Returns:
            sivec: col-major ndarray of shape (ncols, nrows, nroots_si)
                SI vectors with the first dimension iterating over states of fragments not in
                inv and the second dimension iterating over states of fragments in inv 
        '''
        axesorder = [i for i in range (self.nfrags) if not (i in inv)] + list (inv) + [self.nfrags,]
        sivec = self.get_single_rootspace_sivec (iroot).transpose (*axesorder)
        nprods = np.prod (self.lroots[:,iroot])
        nrows = np.prod (self.lroots[inv,iroot])
        ncols = nprods // nrows
        return np.asfortranarray (sivec).reshape ((ncols, nrows, self.nroots_si), order='F')

    def get_fdm (self, rbra, rket, *inv):
        '''Get the n-fragment density matrices for the fragments identified by inv in the bra and
        spaces given by rbra and rket, summing over nonunique excitations

        Args: 
            rbra: integer
                Index of bra rootspace for which to prepare the current cache.
            rket: integer
                Index of ket rootspace for which to prepare the current cache.
            *inv: integers 
                Indices of nonspectator fragments

        Returns:
            fdm : col-major ndarray of shape (lroots[inv[0],rbra], lroots[inv[0],rket],
                                              lroots[inv[1],rbra], lroots[inv[1],rket],
                                               ..., nroots_si)
                len(inv)-fragment reduced density matrix
        '''
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        key = tuple ((rbra,rket)) + tuple (inv)
        braket_table = self.nonuniq_exc[key]
        invset = set ()
        inv = [i for i in inv if not (i in invset or invset.add (i))]
        # must eliminate duplicates, but must also preserve order
        fdm = 0
        for rbra1, rket1 in braket_table:
            b, k, o = self._get_spec_addr_ovlp_1space (rbra1, rket1, *inv)
            # Numpy pads array dimension to the left, so transpose
            sibra = self.get_frag_transposed_sivec (rbra1, *inv)[b,:,:].T * o
            siket = self.get_frag_transposed_sivec (rket1, *inv)[k,:,:].T
            fdm += np.stack ([np.dot (b, k.T) for b, k in zip (sibra, siket)], axis=-1)
        fdm = np.asfortranarray (fdm)
        newshape = list (self.lroots[inv,rbra]) + list (self.lroots[inv,rket]) + [self.nroots_si,]
        fdm = fdm.reshape (newshape, order='F')
        axesorder = sum ([[i, i+len(inv)] for i in range (len(inv))], []) + [2*len(inv),]
        fdm = fdm.transpose (*axesorder)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_o, self.dw_o = self.dt_o + dt, self.dw_o + dw
        return np.asfortranarray (fdm)

    def _get_spec_addr_ovlp_1space (self, rbra, rket, *inv):
        '''Obtain the integer indices and overlap*permutation factors for all pairs of model states
        in the same rootspaces as bra, ket for which a specified list of nonspectator fragments are
        also in same state that they are in a provided input pair bra, ket.

        Args:
            rbra: integer
                Index of a rootspace
            rket: integer
                Index of a rootspace
            *inv: integers
                Indices of nonspectator fragments.

        Returns:
            bra_rng: ndarray of integers
                Indices corresponding to nonzero overlap factors for the ENVs of inv only
            ket_rng: ndarray of integers
                Indices corresponding to nonzero overlap factors for the ENVs of inv only
            o: ndarray of floats
                Overlap * permutation factors (cf. get_ovlp_fac) corresponding to the interactions
                bra_rng, ket_rng.
        '''
        inv = list (set (inv))
        fac = self.spin_shuffle[rbra] * self.spin_shuffle[rket]
        fac *= fermion_frag_shuffle (self.nelec_rf[rbra], inv)
        fac *= fermion_frag_shuffle (self.nelec_rf[rket], inv)
        spec = np.ones (self.nfrags, dtype=bool)
        for i in inv: spec[i] = False
        spec = np.where (spec)[0]
        specints = [self.ints[i] for i in spec]
        o = fac * np.ones ((1,1), dtype=self.dtype)
        for i in specints:
            b, k = i.unique_root[rbra], i.unique_root[rket]
            o = np.multiply.outer (i.ovlp[b][k], o).transpose (0,2,1,3)
            o = o.reshape (o.shape[0]*o.shape[1], o.shape[2]*o.shape[3])
        idx = np.abs(o) > 1e-8
        if self._lowertri and (rbra==rket): # not bra==ket because _loop_lroots_ doesn't restrict to tril
            o[np.diag_indices_from (o)] *= 0.5
            idx[np.triu_indices_from (idx, k=1)] = False
        o = o[idx]
        bra_rng, ket_rng = np.where (idx)
        return bra_rng, ket_rng, o
                
    def _crunch_1d_(self, bra, ket, i):
        '''Compute a single-fragment density fluctuation, for both the 1- and 2-RDMs.'''
        fdm = self.get_fdm (bra, ket, i) # time-profiled by itself
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        inti = self.ints[i]
        d1 = self._get_D1_(bra, ket)
        d2 = self._get_D2_(bra, ket)
        p, q = self.get_range (i)
        d1_s_ii = np.tensordot (fdm, inti.get_dm1 (bra, ket),
                                axes=((0,1),(0,1)))
        d1[:,:,p:q,p:q] = d1_s_ii
        d2_s_iiii = np.tensordot (fdm, inti.get_dm2 (bra, ket),
                                  axes=((0,1),(0,1)))
        d2[:,:,p:q,p:q,p:q,p:q] = d2_s_iiii
        self._put_D1_()
        self._put_D2_()
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_1d, self.dw_1d = self.dt_1d + dt, self.dw_1d + dw

    def _crunch_2d_(self, bra, ket, i, j):
        '''Compute a two-fragment density fluctuation.'''
        fdm = self.get_fdm (bra, ket, i, j) # time-profiled by itself
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        d2 = self._get_D2_(bra, ket)
        inti, intj = self.ints[i], self.ints[j]
        p, q = self.get_range (i)
        r, s = self.get_range (j)
        # d2_IIrJJ
        d2_rsiijj = fdm.transpose (0,1,4,2,3)
        # d2_rsiiJJ
        d2_rsiijj = np.tensordot (inti.get_dm1 (bra, ket), d2_rsiijj,
                                  axes=((0,1),(0,1))).transpose (3,0,1,2,4,5)
        # d2_rsiisjj
        d2_rsiijj = np.tensordot (d2_rsiijj, intj.get_dm1 (bra, ket), axes=2)
        # d2_rssiijj
        d2_rsiijj = d2_rsiijj.transpose (0,1,4,2,3,5,6)
        d2_rsiijj = d2_rsiijj.reshape (self.nroots_si, 4, q-p, q-p, s-r, s-r)
        d2[:,:,p:q,p:q,r:s,r:s] = d2_rsiijj
        d2[:,(0,3),r:s,r:s,p:q,p:q] = d2_rsiijj[:,(0,3),...].transpose (0,1,4,5,2,3)
        d2[:,(1,2),r:s,r:s,p:q,p:q] = d2_rsiijj[:,(2,1),...].transpose (0,1,4,5,2,3)
        d2[:,(0,3),p:q,r:s,r:s,p:q] = -d2_rsiijj[:,(0,3),...].transpose (0,1,2,5,4,3)
        d2[:,(0,3),r:s,p:q,p:q,r:s] = -d2_rsiijj[:,(0,3),...].transpose (0,1,4,3,2,5)
        self._put_D2_()
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_2d, self.dw_2d = self.dt_2d + dt, self.dw_2d + dw

    def _crunch_1c_(self, bra, ket, i, j, s1):
        '''Compute the reduced density matrix elements of a single electron hop; i.e.,
    
        <bra|j'(s1)i(s1)|ket>
    
        i.e.,
    
        j ---s1---> i
    
        and conjugate transpose
        '''
        fdm = self.get_fdm (bra, ket, i, j) # time-profiled by itself
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
        d1_IIrJJ = fdm.transpose (0,1,4,2,3)
        d1_riJJ = np.tensordot (inti.get_p (bra, ket, s1), d1_IIrJJ,
                                axes=((0,1),(0,1))).transpose (1,0,2,3)
        d1_rij = np.tensordot (d1_riJJ, intj.get_h (bra, ket, s1), axes=2)
        d1[:,s1,p:q,r:s] = fac * d1_rij
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
        d2_risjjj = np.tensordot (d1_riJJ, intj.get_phh (bra, ket, s1), axes=2)
        d2_rsijjj = fac * d2_risjjj.transpose (0,2,1,5,3,4)
        _crunch_1c_tdm2 (d2_rsijjj, p, q, r, s, r, s)
        # pph (transpose from Dirac order to Mulliken order)
        d2_rsiiiJJ = np.tensordot (inti.get_pph (bra, ket, s1), d1_IIrJJ,
                                   axes=((0,1),(0,1))).transpose (4,0,1,2,3,5,6)
        d2_rsiiij = np.tensordot (d2_rsiiiJJ, intj.get_h (bra, ket, s1), axes=2)
        d2_rsijii = fac * d2_rsiiij.transpose (0,1,2,5,3,4)
        _crunch_1c_tdm2 (d2_rsijii, p, q, r, s, p, q)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_1c, self.dw_1c = self.dt_1c + dt, self.dw_1c + dw
        self._put_D1_()
        self._put_D2_()

    def _crunch_1s_(self, bra, ket, i, j):
        '''Compute the reduced density matrix elements of a spin unit hop; i.e.,

        <bra|i'(a)j'(b)i(b)j(a)|ket>

        i.e.,

        j ---a---> i
        i ---b---> j

        and conjugate transpose
        '''
        fdm = self.get_fdm (bra, ket, i, j) # time-profiled by itself
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        inti, intj = self.ints[i], self.ints[j]
        d2 = self._get_D2_(bra, ket) # aa, ab, ba, bb -> 0, 1, 2, 3
        p, q = self.get_range (i)
        r, s = self.get_range (j)
        y, z = min (i, j), max (i, j)
        fac = -1
        d2_IIrJJ = fdm.transpose (0,1,4,2,3)
        d2_riiJJ = np.tensordot (inti.get_sp (bra, ket), d2_IIrJJ,
                                 axes=((0,1),(0,1))).transpose (2,0,1,3,4)
        d2_riijj = fac * np.tensordot (d2_riiJJ, intj.get_sm (bra, ket), axes=2)
        d2[:,1,p:q,r:s,r:s,p:q] = d2_riijj.transpose (0,1,4,3,2)
        d2[:,2,r:s,p:q,p:q,r:s] = d2_riijj.transpose (0,3,2,1,4)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_1s, self.dw_1s = self.dt_1s + dt, self.dw_1s + dw
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
        fdm = self.get_fdm (bra, ket, i, j, k, l) # time-profiled by itself
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
        if i == k:
            axesorder = [0,1,fdm.ndim-1] + list (range (2,fdm.ndim-1))
            d2_IIrJJ = fdm.transpose (*axesorder)
            d2_iirJJ = np.tensordot (self.ints[i].get_pp (bra, ket, s2lt), d2_IIrJJ,
                                     axes=((0,1),(0,1)))
            axesorder = [2,0,1] + list (range (3,fdm.ndim))
            d2_riiJJ = d2_iirJJ.transpose (*axesorder)
        else:
            assert (False) # TODO
            pp = np.multiply.outer (self.ints[i].get_1_p (bra, ket, s11),
                                    self.ints[k].get_1_p (bra, ket, s12))
            fac *= (1,-1)[int (i>k)]
            fac *= fermion_des_shuffle (nelec_f_bra, (i, j, k, l), i)
            fac *= fermion_des_shuffle (nelec_f_bra, (i, j, k, l), k)
        if j == l:
            d2_riijj = np.tensordot (d2_riiJJ, self.ints[j].get_hh (bra, ket, s2lt), axes=2)
        else:
            assert (False) # TODO
            hh = np.multiply.outer (self.ints[l].get_1_h (bra, ket, s12),
                                    self.ints[j].get_1_h (bra, ket, s11))
            fac *= (1,-1)[int (j>l)]
            fac *= fermion_des_shuffle (nelec_f_ket, (i, j, k, l), j)
            fac *= fermion_des_shuffle (nelec_f_ket, (i, j, k, l), l)
        d2_rijkl = fac * d2_riijj.transpose (0,1,4,2,3) # Dirac -> Mulliken transp
        p, q = self.get_range (i)
        r, s = self.get_range (j)
        t, u = self.get_range (k)
        v, w = self.get_range (l)
        d2[:,s2, p:q,r:s,t:u,v:w] = d2_rijkl
        d2[:,s2T,t:u,v:w,p:q,r:s] = d2_rijkl.transpose (0,3,4,1,2)
        if s2 == s2T: # same-spin only: exchange happens
            d2[:,s2,p:q,v:w,t:u,r:s] = -d2_rijkl.transpose (0,1,4,3,2)
            d2[:,s2,t:u,r:s,p:q,v:w] = -d2_rijkl.transpose (0,3,2,1,4)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_2c, self.dw_2c = self.dt_2c + dt, self.dw_2c + dw
        self._put_D2_()

    def _put_D1_(self):
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        fn = liblassi.LASSIRDMdputSD1
        fn (self._rdm1s_c, self._d1buf_c,
            self._si_c_ncol, self._norb_c, self._nsrc_c,
            self._dblk_idx, self._sblk_idx, self._lblk, self._nblk)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_p, self.dw_p = self.dt_p + dt, self.dw_p + dw

    def _put_D2_(self):
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        fn = liblassi.LASSIRDMdputSD2
        fn (self._rdm2s_c, self._d2buf_c,
            self._si_c_ncol, self._norb_c, self._nsrc_c, self._pdest,
            self._dblk_idx, self._sblk_idx, self._lblk, self._nblk)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_p, self.dw_s = self.dt_p + dt, self.dw_s + dw

    def _crunch_env_(self, _crunch_fn, *row):
        if _crunch_fn.__name__ in ('_crunch_1c_', '_crunch_1c1d_', '_crunch_2c_'):
            inv = row[2:-1]
        else:
            inv = row[2:]
        with lib.temporary_env (self, **self._orbrange_env_kwargs (inv)):
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
    log = logger.new_logger (las, las.verbose)
    nlas = las.ncas_sub
    ncas = las.ncas 
    nroots_si = si.shape[-1]
    max_memory = getattr (las, 'max_memory', las.mol.max_memory)
    dtype = ci[0][0].dtype 
        
    # First pass: single-fragment intermediates
    hopping_index, ints, lroots = op_o1.make_ints (las, ci, nelec_frs)
    nstates = np.sum (np.prod (lroots, axis=0))
        
    # Second pass: upper-triangle
    outerprod = LRRDMint (ints, nlas, hopping_index, lroots, si, dtype=dtype,
                          max_memory=max_memory, log=log)

    # Spoof nonuniq_exc to avoid summing together things that need to be separate
    for iroot in range (outerprod.nroots):
        val = [[iroot,iroot]]
        for ifrag in range (outerprod.nfrags):
            key = (iroot, iroot, ifrag)
            outerprod.nonuniq_exc[key] = val
    outerprod._lowertri = False
    def make_fdm1 (iroot, ifrag):
        fdm = outerprod.get_fdm (iroot, iroot, ifrag).transpose (2,0,1)
        if iroot in ints[ifrag].umat_root:
            umat = ints[ifrag].umat_root[iroot]
            fdm = lib.einsum ('rij,ik,jl->rkl',fdm,umat,umat)
        return fdm
    return make_fdm1

def roots_make_rdm12s (las, ci, nelec_frs, si, **kwargs):
    return op_o1.roots_make_rdm12s (las, ci, nelec_frs, si, _LRRDMint_class=LRRDMint, **kwargs)

        

