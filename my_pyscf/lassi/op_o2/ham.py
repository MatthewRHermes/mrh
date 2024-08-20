import numpy as np
from mrh.my_pyscf.lassi import op_o1
from pyscf import lib
from pyscf.lib import logger
from itertools import product

fermion_frag_shuffle = op_o1.fermion_frag_shuffle
fermion_des_shuffle = op_o1.fermion_des_shuffle

# S2 is:
# (d2aa_ppqq + d2bb_ppqq - d2ab_ppqq - d2ba_ppqq)/4 - (d2ab_pqqp + d2ba_pqqp)/2
# for two fragments this is (sign?)
# ((d1a_pp - d1b_pp) * (d1a_qq - d1b_qq))/4 - (sp_pp*sm_qq + sm_pp*sp_qq)/2

class HamS2ovlpint (op_o1.HamS2ovlpint):
    ___doc__ = op_o1.HamS2ovlpint.__doc__ + '''

    SUBCLASS: vectorize lroots

    '''

    def canonical_operator_order (self, arr, inv):
        assert ((arr.ndim % 2) == 0)
        if arr.ndim < 3: return arr
        invset = set ()
        inv = [i for i in inv if not (i in invset or invset.add (i))]
        ndim = arr.ndim // 2
        # sort by fragments
        arr_shape = np.asarray (arr.shape).reshape (ndim, 2)
        arr = arr.reshape (np.prod (arr_shape, axis=1))
        inv = np.asarray (inv)
        idx = list (np.argsort (-inv))
        arr = arr.transpose (*idx)
        arr_shape = arr_shape[idx,:]
        arr_shape = arr_shape.ravel ()
        arr = arr.reshape (arr_shape)
        # ...ckbjai -> ...cba...kji
        axesorder = np.arange (arr.ndim, dtype=int).reshape (ndim, 2)
        axesorder = np.ravel (axesorder.T)
        arr = arr.transpose (*axesorder)
        return arr

    def _crunch_env_(self, _crunch_fn, *row):
        if _crunch_fn.__name__ in ('_crunch_1c_', '_crunch_1c1d_', '_crunch_2c_'):
            inv = row[2:-1]
        else:
            inv = row[2:]
        with lib.temporary_env (self, **self._orbrange_env_kwargs (inv)):
            self._prepare_spec_addr_ovlp_(row[0], row[1], *inv)
            ham, s2, ninv = _crunch_fn (*row)
            ham = self.canonical_operator_order (ham, ninv)
            s2 = self.canonical_operator_order (s2, ninv)
            self._put_ham_s2_(row[0], row[1], ham, s2, *inv)

    _put_ham_s2_1state_ = op_o1.HamS2ovlpint._put_ham_s2_

    def _put_ham_s2_(self, bra, ket, ham, s2, *inv):
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        # TODO: vectorize this part
        bra_rng = self._get_addr_range (bra, *inv)
        ket_rng = self._get_addr_range (ket, *inv)
        ham = ham.ravel ()
        s2 = s2.ravel ()
        for ham_bk, s2_bk, (bra1, ket1) in zip (ham, s2, product (bra_rng, ket_rng)):
            bra2, ket2, wgt = self._get_spec_addr_ovlp (bra1, ket1, *inv)
            self._put_ham_s2_1state_(bra2, ket2, ham_bk, s2_bk, wgt)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_p, self.dw_p = self.dt_p + dt, self.dw_p + dw

    def get_ham_2q (self, *inv):
        assert (len (inv) in (2,4))
        i, j = inv[:2]
        p, q = self.get_range (i)
        r, s = self.get_range (j)
        if len (inv) == 2:
            return self.h1[:,p:q,r:s]
        k, l = inv[2:]
        t, u = self.get_range (k)
        v, w = self.get_range (l)
        return self.h2[p:q,r:s,t:u,v:w]

    def _crunch_1d_(self, bra, ket, i):
        '''Compute a single-fragment density fluctuation, for both the 1- and 2-RDMs.'''
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        inti = self.ints[i]
        d1s = inti.get_dm1 (bra, ket)
        d2s = inti.get_dm2 (bra, ket)
        d2 = d2s.sum (2)
        ham  = np.tensordot (d1s, self.get_ham_2q (i,i), axes=3)
        ham += np.tensordot (d2, self.get_ham_2q (i,i,i,i), axes=4) * .5
        s2 = 3 * np.trace (d1s, axis1=-2, axis2=-1).sum (-1) / 4
        m2 = d2s.diagonal (axis1=3,axis2=4).diagonal (axis1=3, axis2=4).sum ((3,4)) / 4
        m2 = m2.transpose (2,0,1)
        d2s = d2s.transpose (2,0,1,3,4,5,6)
        s2 += m2[0] + m2[3] - m2[1] - m2[2]
        m2 = (d2s[1]+d2s[2]).diagonal (axis1=2, axis2=5).diagonal (axis1=2,axis2=3)
        s2 -= m2.sum ((2,3)) / 2
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_1d, self.dw_1d = self.dt_1d + dt, self.dw_1d + dw
        return ham, s2, (i)
    
    def _crunch_2d_(self, bra, ket, i, j):
        '''Compute a two-fragment density fluctuation.'''
        # 1/2 factor of h2 canceled by i<->j
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        d1s_ii = self.ints[i].get_dm1 (bra, ket)
        d1s_jj = self.ints[j].get_dm1 (bra, ket)
        h_ = self.get_ham_2q (j,j,i,i)
        h_ = np.tensordot (d1s_ii.sum (2), h_, axes=((-2,-1),(-2,-1)))
        h_ = np.tensordot (d1s_jj.sum (2), h_, axes=((-2,-1),(-2,-1)))
        ham = h_
        hs_ = self.get_ham_2q (j,i,i,j).transpose (0,3,2,1)
        hs_ = np.tensordot (d1s_ii, hs_, axes=((-2,-1),(-2,-1)))
        hs_ = hs_.transpose (2,0,1,3,4)
        d1s_ii = d1s_ii.transpose (2,0,1,3,4)
        d1s_jj = d1s_jj.transpose (2,0,1,3,4)
        for h_, d1_jj in zip (hs_, d1s_jj):
            ham -= np.tensordot (d1_jj, h_, axes=((-2,-1),(-2,-1)))
        mi = np.trace (d1s_ii, axis1=-2, axis2=-1)
        mi = (mi[0] - mi[1])
        mj = np.trace (d1s_jj, axis1=-2, axis2=-1)
        mj = (mj[0] - mj[1]) 
        s2 = np.multiply.outer (mj, mi) / 2
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_2d, self.dw_2d = self.dt_2d + dt, self.dw_2d + dw
        return ham, s2, (j, i)

    def _crunch_1c_(self, bra, ket, i, j, s1):
        '''Compute the reduced density matrix elements of a single electron hop; i.e.,
        
        <bra|j'(s1)i(s1)|ket>
            
        i.e.,
            
        j ---s1---> i

        and conjugate transpose
        '''
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        fac = 1 # h1 involved, so no 1/2 here
        nelec_f_bra = self.nelec_rf[bra]
        nelec_f_ket = self.nelec_rf[ket]
        fac *= fermion_des_shuffle (nelec_f_bra, (i, j), i)
        fac *= fermion_des_shuffle (nelec_f_ket, (i, j), j)
        h_ = self.get_ham_2q (j,i)[s1] # BEWARE CONJ
        p_i = self.ints[i].get_p (bra, ket, s1)
        pph_i = self.ints[i].get_pph (bra, ket, s1).sum (2)
        h_j = self.ints[j].get_h (bra, ket, s1)
        phh_j = self.ints[j].get_phh (bra, ket, s1).sum (2)
        h_ = np.tensordot (p_i, h_, axes=((-1),(-1)))
        h_ = np.tensordot (h_j, h_, axes=((-1),(-1)))
        ham = h_
        h_jiii = self.get_ham_2q (j,i,i,i) # BEWARE CONJ
        h_ = np.tensordot (pph_i, h_jiii, axes=((-3,-2,-1),(-3,-1,-2)))
        ham += np.tensordot (h_j, h_, axes=((-1),(-1)))
        h_ = self.get_ham_2q (j,j,j,i) # BEWARE CONJ
        h_ = np.tensordot (p_i, h_, axes=((-1),(-1)))
        ham += np.tensordot (phh_j, h_, axes=((-3,-2,-1),(-2,-3,-1)))
        ham *= fac
        s2 = np.zeros_like (ham)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_1c, self.dw_1c = self.dt_1c + dt, self.dw_1c + dw
        return ham, s2, (j, i)

    def _crunch_1c1d_(self, bra, ket, i, j, k, s1):
        '''Compute the reduced density matrix elements of a coupled electron-hop and
        density fluctuation.'''
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        fac = 1 # 1/2 factor of h2 canceled by ijkk, ikkj <-> kkij, kjik
        nelec_f_bra = self.nelec_rf[bra]
        nelec_f_ket = self.nelec_rf[ket]
        fac *= fermion_des_shuffle (nelec_f_bra, (i, j, k), i)
        fac *= fermion_des_shuffle (nelec_f_ket, (i, j, k), j)
        p_i = self.ints[i].get_p (bra, ket, s1)
        h_j = self.ints[j].get_h (bra, ket, s1)
        d1s_k = self.ints[k].get_dm1 (bra, ket)
        d1_k = d1s_k.sum (2)
        h_ = self.get_ham_2q (k,k,j,i) # BEWARE CONJ
        h_ = np.tensordot (p_i, h_, axes=((-1),(-1)))
        h_ = np.tensordot (h_j, h_, axes=((-1),(-1)))
        ham = np.tensordot (d1_k, h_, axes=((-2,-1),(-2,-1)))
        d1_k = d1s_k[:,:,s1]
        h_ = self.get_ham_2q (j,k,k,i).transpose (1,2,0,3) # BEWARE CONJ
        h_ = np.tensordot (p_i, h_, axes=((-1),(-1)))
        h_ = np.tensordot (h_j, h_, axes=((-1),(-1)))
        ham -= np.tensordot (d1_k, h_, axes=((-2,-1),(-2,-1)))
        ham *= fac
        s2 = np.zeros_like (ham)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_1c1d, self.dw_1c1d = self.dt_1c1d + dt, self.dw_1c1d + dw
        return ham, s2, (k, j, i)

    def _crunch_1s_(self, bra, ket, i, j):
        '''Compute the reduced density matrix elements of a spin unit hop; i.e.,

        <bra|i'(a)j'(b)i(b)j(a)|ket>

        i.e.,

        j ---a---> i
        i ---b---> j

        and conjugate transpose
        '''
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        fac = -1 # 1/2 factor of h2 canceled by i<->j
        sp_i = self.ints[i].get_sp (bra, ket)
        sm_j = self.ints[j].get_sm (bra, ket)
        ham = self.get_ham_2q (j,i,i,j).transpose (0,3,2,1)
        ham = np.tensordot (sp_i, ham, axes=((-2,-1),(-2,-1)))
        ham = np.tensordot (sm_j, ham, axes=((-2,-1),(-2,-1)))
        ham *= fac
        sp_i = np.trace (sp_i, axis1=-2, axis2=-1)
        sm_j = np.trace (sm_j, axis1=-2, axis2=-1)
        s2 = -fac * np.multiply.outer (sm_j, sp_i)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_1s, self.dw_1s = self.dt_1s + dt, self.dw_1s + dw
        return ham, s2, (j, i)

    def _crunch_1s1c_(self, bra, ket, i, j, k):
        '''Compute the reduced density matrix elements of a spin-charge unit hop; i.e.,

        <bra|i'(a)k'(b)j(b)k(a)|ket>

        i.e.,

        k ---a---> i
        j ---b---> k

        and conjugate transpose
        '''
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        fac = -1 # a'bb'a -> a'ab'b sign; 1/2 factor of h2 canceled by ikkj<->kjik
        nelec_f_bra = self.nelec_rf[bra]
        nelec_f_ket = self.nelec_rf[ket]
        fac *= fermion_des_shuffle (nelec_f_bra, (i, j, k), i)
        fac *= fermion_des_shuffle (nelec_f_ket, (i, j, k), j)
        ham = self.get_ham_2q (j,k,k,i).transpose (1,2,0,3) # BEWARE CONJ
        ham = np.tensordot (self.ints[i].get_p (bra, ket, 0), ham, axes=((-1),(-1)))
        ham = np.tensordot (self.ints[j].get_h (bra, ket, 1), ham, axes=((-1),(-1)))
        ham = np.tensordot (self.ints[k].get_sm (bra, ket), ham, axes=((-2,-1),(-2,-1)))
        ham *= fac
        s2 = np.zeros_like (ham)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_1s1c, self.dw_1s1c = self.dt_1s1c + dt, self.dw_1s1c + dw
        return ham, s2, (k, j, i)

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
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        # s2lt: 0, 1, 2 -> aa, ab, bb
        # s2: 0, 1, 2, 3 -> aa, ab, ba, bb
        s2  = (0, 1, 3)[s2lt] # aa, ab, bb
        s2T = (0, 2, 3)[s2lt] # aa, ba, bb -> when you populate the e1 <-> e2 permutation
        s11 = s2 // 2
        s12 = s2 % 2
        nelec_f_bra = self.nelec_rf[bra]
        nelec_f_ket = self.nelec_rf[ket]
        fac = (1,.5)[int ((i,j,s11)==(k,l,s12))] # 1/2 factor of h2 canceled by ijkl <-> klij
        ham = self.get_ham_2q (l,k,j,i).transpose (0,2,3,1) # BEWARE CONJ
        if s11==s12 and i!=k and j!=l: # exchange
            ham -= self.get_ham_2q (l,i,j,k).transpose (0,2,1,3) # BEWARE CONJ
        if i == k:
            ham = np.tensordot (self.ints[i].get_pp (bra, ket, s2lt), ham, axes=((-2,-1),(-2,-1)))
        else:
            ham = np.tensordot (self.ints[k].get_p (bra, ket, s12), ham, axes=((-1),(-1)))
            ham = np.tensordot (self.ints[i].get_p (bra, ket, s11), ham, axes=((-1),(-1)))
            fac *= (1,-1)[int (i>k)]
            fac *= fermion_des_shuffle (nelec_f_bra, (i, j, k, l), i)
            fac *= fermion_des_shuffle (nelec_f_bra, (i, j, k, l), k)
        if j == l:
            ham = np.tensordot (self.ints[j].get_hh (bra, ket, s2lt), ham, axes=((-2,-1),(-2,-1)))
        else:
            ham = np.tensordot (self.ints[j].get_h (bra, ket, s11), ham, axes=((-1),(-1)))
            ham = np.tensordot (self.ints[l].get_h (bra, ket, s12), ham, axes=((-1),(-1)))
            fac *= (1,-1)[int (j>l)]
            fac *= fermion_des_shuffle (nelec_f_ket, (i, j, k, l), j)
            fac *= fermion_des_shuffle (nelec_f_ket, (i, j, k, l), l)
        ham *= fac
        s2 = np.zeros_like (ham)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_2c, self.dw_2c = self.dt_2c + dt, self.dw_2c + dw
        return ham, s2, (l, j, i, k)

#ham = op_o1.ham
def ham (las, h1, h2, ci, nelec_frs, **kwargs):
    return op_o1.ham (las, h1, h2, ci, nelec_frs,
                      _HamS2ovlpint_class=HamS2ovlpint)


