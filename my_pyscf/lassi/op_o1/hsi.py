import numpy as np
from scipy.sparse import linalg as sparse_linalg
from pyscf import lib
from pyscf.lib import logger, param
from mrh.my_pyscf.lassi import citools
from mrh.my_pyscf.lassi.op_o1 import frag, opterm
from mrh.my_pyscf.lassi.op_o1.rdm import LRRDM
from mrh.my_pyscf.lassi.op_o1.hams2ovlp import HamS2Ovlp, ham, soc_context
from mrh.my_pyscf.lassi.citools import _fake_gen_contract_op_si_hdiag
from mrh.my_pyscf.lassi.op_o1.utilities import *
import functools
from itertools import product
from pyscf import __config__
import sys

PROFVERBOSE = getattr (__config__, 'lassi_hsi_profverbose', None)
SCREEN_THRESH = getattr (__config__, 'lassi_hsi_screen_thresh', 1e-12)

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
    def __init__(self, ints, nlas, lroots, h1, h2, mask_bra_space=None,
                 mask_ket_space=None, pt_order=None, do_pt_order=None, log=None,
                 max_memory=param.MAX_MEMORY, screen_thresh=SCREEN_THRESH, dtype=np.float64):
        t0 = (logger.process_clock (), logger.perf_counter ())
        HamS2Ovlp.__init__(self, ints, nlas, lroots, h1, h2,
                           mask_bra_space=mask_bra_space, mask_ket_space=mask_ket_space,
                           pt_order=pt_order, do_pt_order=do_pt_order,
                           log=log, max_memory=max_memory, dtype=dtype)
        self.log = logger.new_logger (self.log, verbose=PROFVERBOSE)
        self.screen_thresh = screen_thresh
        self.x = self.si = np.zeros (self.nstates, self.dtype)
        self.ox = np.zeros (self.nstates, self.dtype)
        self.ox1 = np.zeros (self.nstates, self.dtype)
        self.init_cache_profiling ()
        self.checkmem_oppart ()
        self._cache_()

    def checkmem_oppart (self):
        rm = 0
        for exc, fn in zip ((self.exc_1d, self.exc_2d, self.exc_1s, self.exc_1c, self.exc_1c1d,
                             self.exc_1s1c, self.exc_2c),
                            (self._crunch_1d_, self._crunch_2d_, self._crunch_1s_,
                             self._crunch_1c_, self._crunch_1c1d_, self._crunch_1s1c_,
                             self._crunch_2c_)):
            rm += self.checkmem_1oppart (exc, fn)
        m0 = lib.current_memory ()[0]
        memstr = "hsi operator cache req's >= {} MB ({} MB current; {} MB available)".format (
            rm, m0, self.max_memory)
        self.log.debug (memstr)
        if (m0 + rm) > self.max_memory:
            raise MemoryError (memstr)

    def checkmem_1oppart (self, exc, fn):
        rm = 0
        has_s = self._fn_contributes_to_s2 (fn)
        def maxabs (op):
            if callable (getattr (op, 'maxabs', None)):
                return op.maxabs ()
            else:
                return np.amax (np.abs (op))
        for row in exc:
            if self._fn_row_has_spin (fn):
                inv = row[2:-1]
            else:
                inv = row[2:]
            bra, ket = row[:2]
            inv = list (set (inv))
            if len (inv) == 4:
                data = fn (*row, dry_run=True)
                if maxabs (data[0]) >= self.screen_thresh:
                    rm += data[0].size
                if data[1] is not None and (maxabs (data[1]) >= self.screen_thresh):
                    rm += data[1].size
            else:
                opbralen = np.prod (self.lroots[inv,bra])
                opketlen = np.prod (self.lroots[inv,ket])
                rm += (1 + int (has_s)) * opbralen * opketlen
        rm *= self.dtype.itemsize / 1e6
        self.log.debug ("{} op cache req's {} MB".format (fn.__name__, rm))
        return rm

    def _cache_(self):
        t0 = (logger.process_clock (), logger.perf_counter ())
        self.optermgroups_s = {}
        self.optermgroups_h = {}
        for exc, fn in zip ((self.exc_1d, self.exc_2d, self.exc_1s, self.exc_1c, self.exc_1c1d,
                             self.exc_1s1c, self.exc_2c),
                            (self._crunch_1d_, self._crunch_2d_, self._crunch_1s_,
                             self._crunch_1c_, self._crunch_1c1d_, self._crunch_1s1c_,
                             self._crunch_2c_)):
            self._crunch_oppart_(exc, fn)
        self.optermgroups_s = self._index_ovlppart (self.optermgroups_s)
        self.optermgroups_h = self._index_ovlppart (self.optermgroups_h)
        self.log.debug (self.sprint_cache_profile ())
        self.log.timer ('HamS2OvlpOperators operator cacheing', *t0)
        if self.log.verbose >= logger.DEBUG:
            sizes = self.get_dot_product_sizes (self.optermgroups_h)
            self.log.debug ('Hamiltonian operator cost estimator function by # frags:')
            self.log.debug (' %6s %8s %8s', 'nfrags', 'cost', 'nvecs')
            for i, arr in enumerate (sizes):
                mysize = np.prod (arr, axis=1).astype (float).sum ()
                nvecs = arr[:,2].astype (float).sum ()
                self.log.debug (' %6d %8.1e %8.1e', i+1, mysize, nvecs)

    def get_dot_product_sizes (self, groups):
        sizes = [{} for i in range (4)]
        for inv, group in groups.items ():
            sinv = list (set (inv))
            nfrags = len (sinv)
            for op in group.ops:
                K, L = op.get_formal_shape ()
                M = sizes[nfrags-1].get ((K,L), 0)
                Mt = sizes[nfrags-1].get ((L,K), 0)
                for key in op.spincase_keys:
                    brakets, bras, braHs = self.get_nonuniq_exc_square (key)
                    for bra in bras:
                        urootstr = self.urootstr[:,bra].copy ()
                        urootstr[sinv] = 1
                        M += np.prod (urootstr)
                    for bra in braHs:
                        urootstr = self.urootstr[:,bra].copy ()
                        urootstr[sinv] = 1
                        Mt += np.prod (urootstr)
                sizes[nfrags-1][(K,L)] = M
                if Mt > 0:
                    sizes[nfrags-1][(L,K)] = Mt
        for i in range (4):
            d = sizes[i]
            sizes[i] = np.zeros ((len(d),3), dtype=int)
            for j, ((K, L), M) in enumerate (d.items ()):
                sizes[i][j,:] = [K, L, M]
        return sizes

    def opterm_std_shape (self, bra, ket, op, inv, sinv):
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        if isinstance (op, (np.ndarray, opterm.OpTerm)):
            op = self.canonical_operator_order (op, sinv)
            opbralen = np.prod (self.lroots[inv,bra])
            opketlen = np.prod (self.lroots[inv,ket])
            op = op.reshape ((opbralen, opketlen), order='C')
            op = opterm.as_opterm (op)
        t1, w1 = logger.process_clock (), logger.perf_counter ()
        self.dt_oT += (t1-t0)
        self.dw_oT += (w1-w0)
        return op

    def _crunch_oppart_(self, exc, fn):
        has_s = self._fn_contributes_to_s2 (fn)
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
            op = self.opterm_std_shape (bra, ket, data[0], inv, sinv)
            key = tuple (inv)
            if op.maxabs () >= self.screen_thresh:
                val = self.optermgroups_h.get (key, opterm.OpTermGroup (key))
                for mybra, myket in self.spman[tuple((bra,ket))+tuple(row)]:
                    op.spincase_keys.append (tuple ((mybra, myket)) + tuple (row))
                val.append (op)
                self.optermgroups_h[key] = val
            if has_s:
                op = self.opterm_std_shape (bra, ket, data[1], inv, sinv)
                if op.maxabs () >= self.screen_thresh:
                    val = self.optermgroups_s.get (key, opterm.OpTermGroup (key))
                    for mybra, myket in self.spman[tuple((bra,ket))+tuple(row)]:
                        op.spincase_keys.append (tuple ((mybra, myket)) + tuple (row))
                    val.append (op)
                    self.optermgroups_s[key] = val
            

    def _index_ovlppart (self, groups):
        # TODO: redesign this in a scalable graph-theoretic way
        # TODO: memcheck for this. It's hard b/c IDK how to guess the final size of ovlplinkstr
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        #x0 = (logger.process_clock (), logger.perf_counter ())
        for inv, group in groups.items ():
            len_tab = 0
            for op in group.ops:
                for key in op.spincase_keys:
                    tab = self.nonuniq_exc[key]
                    len_tab += tab.shape[0]
                    len_tab += np.count_nonzero (tab[:,0] != tab[:,1])
            #x0 = self.log.timer ('tab_len', *x0)
            ovlplinkstr = np.empty ((len_tab, self.nfrags+1), dtype=int)
            seen = set ()
            i = 0
            for op in group.ops:
                for key in op.spincase_keys:
                    tab = self.nonuniq_exc[key]
                    for bra, ket in tab:
                        ovlplinkstr[i,0] = ket
                        ovlplinkstr[i,1:] = self.ox_ovlp_urootstr (bra, ket, inv)
                        fp = hash (tuple (ovlplinkstr[i,:]))
                        if fp not in seen:
                            seen.add (fp)
                            i += 1
                        if bra != ket:
                            ovlplinkstr[i,0] = bra
                            ovlplinkstr[i,1:] = self.ox_ovlp_urootstr (ket, bra, inv)
                            fp = hash (tuple (ovlplinkstr[i,:]))
                            if fp not in seen:
                                seen.add (fp)
                                i += 1
            ovlplinkstr = ovlplinkstr[:i]
            group.ovlplink = np.asarray (ovlplinkstr)
        t1, w1 = logger.process_clock (), logger.perf_counter ()
        self.dt_i += (t1-t0)
        self.dw_i += (w1-w0)
        return groups

    def get_nonuniq_exc_square (self, key, also_bras=True):
        tab_bk = self.nonuniq_exc[key]
        idx_equal = tab_bk[:,0]==tab_bk[:,1]
        tab_kb = tab_bk[~idx_equal,::-1]
        brakets = np.append (tab_bk, tab_kb, axis=0)
        if not also_bras: return brakets
        bras = set (tab_bk[:,0])
        braHs = set (tab_kb[:,0]) - bras
        return brakets, list (bras), list (braHs)

    def init_cache_profiling (self):
        self.dt_1d, self.dw_1d = 0.0, 0.0
        self.dt_2d, self.dw_2d = 0.0, 0.0
        self.dt_1c, self.dw_1c = 0.0, 0.0
        self.dt_1c1d, self.dw_1c1d = 0.0, 0.0
        self.dt_1s, self.dw_1s = 0.0, 0.0
        self.dt_1s1c, self.dw_1s1c = 0.0, 0.0
        self.dt_2c, self.dw_2c = 0.0, 0.0
        self.dt_i, self.dw_i = 0.0, 0.0
        self.dt_o, self.dw_o = 0.0, 0.0
        self.dt_oT, self.dw_oT = 0.0, 0.0

    def init_profiling (self):
        self.dt_u, self.dw_u = 0.0, 0.0
        self.dt_sX, self.dw_sX = 0.0, 0.0
        self.dt_oX, self.dw_oX = 0.0, 0.0
        self.dt_pX, self.dw_pX = 0.0, 0.0
        self.dt_4fo, self.dw_4fo = 0.0, 0.0
        self.dt_4fr, self.dw_4fr = 0.0, 0.0
        self.dt_4f1, self.dw_4f1 = 0.0, 0.0
        self.dt_4f2, self.dw_4f2 = 0.0, 0.0
        self.dt_4f3, self.dw_4f3 = 0.0, 0.0
        self.dt_oXn = [0.0, 0.0, 0.0, 0.0]
        self.dw_oXn = [0.0, 0.0, 0.0, 0.0]

    def sprint_cache_profile (self):
        fmt_str = '{:>5s} CPU: {:9.2f} ; wall: {:9.2f}'
        profile = fmt_str.format ('1d', self.dt_1d, self.dw_1d)
        profile += '\n' + fmt_str.format ('2d', self.dt_2d, self.dw_2d)
        profile += '\n' + fmt_str.format ('1c', self.dt_1c, self.dw_1c)
        profile += '\n' + fmt_str.format ('1c1d', self.dt_1c1d, self.dw_1c1d)
        profile += '\n' + fmt_str.format ('1s', self.dt_1s, self.dw_1s)
        profile += '\n' + fmt_str.format ('1s1c', self.dt_1s1c, self.dw_1s1c)
        profile += '\n' + fmt_str.format ('2c', self.dt_2c, self.dw_2c)
        profile += '\n' + fmt_str.format ('idx', self.dt_i, self.dw_i)
        profile += '\n' + fmt_str.format ('opT', self.dt_oT, self.dw_oT)
        return profile

    def sprint_profile (self):
        fmt_str = '{:>5s} CPU: {:9.2f} ; wall: {:9.2f}'
        profile = fmt_str.format ('umat', self.dt_u, self.dw_u)
        profile += '\n' + fmt_str.format ('olpX', self.dt_sX, self.dw_sX)
        profile += '\n' + fmt_str.format ('opX', self.dt_oX, self.dw_oX)
        profile += '\n' + fmt_str.format ('putX', self.dt_pX, self.dw_pX)
        profile += '\n' + 'opX components:'
        for i in range (4):
            profile += '\n' + fmt_str.format ('{}f'.format (i+1),
                                              self.dt_oXn[i],
                                              self.dw_oXn[i])
        profile += '\n' + 'opX 4-fragment components:'
        profile += '\n' + fmt_str.format ('4f_o', self.dt_4fo, self.dw_4fo)
        profile += '\n' + fmt_str.format ('4f_r', self.dt_4fr, self.dw_4fr)
        profile += '\n' + fmt_str.format ('4f_1', self.dt_4f1, self.dw_4f1)
        profile += '\n' + fmt_str.format ('4f_2', self.dt_4f2, self.dw_4f2)
        profile += '\n' + fmt_str.format ('4f_3', self.dt_4f3, self.dw_4f3)
        return profile

    def get_xvec (self, iroot, *inv):
        fac = self.spin_shuffle[iroot] * self.fermion_frag_shuffle (iroot, inv)
        i, j = self.offs_lroots[iroot]
        return fac * self.x[i:j]

    def put_ox1_(self, vec, iroot, *inv):
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
        for inv, group in self.optermgroups_h.items (): self._opuniq_x_group_(inv, group)
        self._umat_linequiv_loop_(1) # U.T @ ox
        self.log.info (self.sprint_profile ())
        self.log.timer ('HamS2OvlpOperators._ham_op', *t0)
        return self.ox.copy ()

    def _s2_op (self, x):
        t0 = (logger.process_clock (), logger.perf_counter ())
        self.init_profiling ()
        self.x[:] = x.flat[:]
        self.ox[:] = 0
        self._umat_linequiv_loop_(0) # U.conj () @ x
        for inv, group in self.optermgroups_s.items (): self._opuniq_x_group_(inv, group)
        self._umat_linequiv_loop_(1) # U.T @ ox
        self.log.info (self.sprint_profile ())
        self.log.timer ('HamS2OvlpOperators._s2_op', *t0)
        return self.ox.copy ()

    def _opuniq_x_group_(self, inv, group):
        '''All unique operations which have a set of nonspectator fragments in common'''
        ops, ovlplink = group.ops, group.ovlplink

        t0, w0 = logger.process_clock (), logger.perf_counter ()
        # Some rootspaces are redundant: same urootstr for different ket indices.
        # Those vector slices must be added, so I can't use dict comprehension.
        vecs = {}
        for ket in set (ovlplink[:,0]):
            key = tuple(self.urootstr[:,ket])  #gets a number 
            vecs[key] = self.get_xvec (ket, *inv).reshape (-1,1) + vecs.get (key, 0) #gets a vector from x, adds to it if already existing
        for ifrag in range (self.nfrags):
            if ifrag in inv:
                # Collect the nonspectator-fragment dimensions on the minor end
                for ket, vec in vecs.items ():
                    lket = self.ints[ifrag].get_lroots_uroot (ket[ifrag])
                    lr = vec.shape[-1]
                    vecs[ket] = vec.reshape (-1,lr*lket) #reshape vec into 2D matrix
            else:
                vecs = self.ox_ovlp_frag (ovlplink, vecs, ifrag) #multiply with ovlp part of interactions
        t1, w1 = logger.process_clock (), logger.perf_counter ()
        self.dt_sX += (t1-t0)
        self.dw_sX += (w1-w0)

        #Have a function that takes the vecs dict, and returns a dictionary that has the same keys and values that are slices of the pinned memory and puts the ovecs values on a pinned memory block exactly once. 
        #ox1 is inside the loop, but it's ..
        #accumulate on a pinned memory, get a dictionary like vecs, that gets you the location
        '''VA: Nov 4, 2025: Thoughts on GPU acceleration
          Input: op, vecs
          Output: ox1
          Several op exists, each must be of shape (m_i, k_i). Corresponding vecs are of shape (n_j, k_i).
          ox1 = \sum_i (op_i * \sum_j (vec_ij)_)
          Thoughts on making it GPU'able. 
            vecs exists as a dictionary. The keys can be moved out, and the values can be moved onto a contiguous massive pinned memory. 
            ox1 is just a big block of memory.  
            use the various indexing functions available to get, for (ij) pair, 
              vec: location, size of vec inside the massive vecs block of memory
              ox1: location, size of result storage inside the massive ox1 block of memory
           Copy back ox1 from pinned to pageable.
          '''
        self.ox1[:] = 0 #of shape nstates
        for op in ops:
            for key in op.spincase_keys:  #spincase_keys is a lookup table
                self._opuniq_x_(op, key[0], key[1], vecs, *key[2:])  #vecs exists as a whole, we are indexing into it. A way to store it on pinned would be good.

        t2, w2 = logger.process_clock (), logger.perf_counter ()
        self.dt_oX += (t2-t1)
        self.dw_oX += (w2-w1)
        self.dt_oXn[len(inv)-1] += (t2-t1)
        self.dw_oXn[len(inv)-1] += (w2-w1)

        for bra in range (self.nroots):
            i, j = self.offs_lroots[bra]
            self.ox[i:j] += transpose_sivec_with_slow_fragments (
                self.ox1[i:j], self.lroots[:,bra], *inv
            )[0]
        t3, w3 = logger.process_clock (), logger.perf_counter ()
        self.dt_pX += (t3-t2)
        self.dw_pX += (w3-w2)

    def _profile_4frag_(self, op):
        if not hasattr (op, 'dt_4fr'): return
        self.dt_4fr += op.dt_4fr
        self.dw_4fr += op.dw_4fr
        self.dt_4f1 += op.dt_4f1
        self.dw_4f1 += op.dw_4f1
        self.dt_4f2 += op.dt_4f2
        self.dw_4f2 += op.dw_4f2
        self.dt_4f3 += op.dt_4f3
        self.dw_4f3 += op.dw_4f3
        return

    def _opuniq_x_(self, op, obra, oket, ovecs, *inv):
        '''All operations which are unique in that a given set of nonspectator fragment bra
        statelets are coupled to a given set of nonspectator fragment ket statelets'''
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        op = opterm.reduce_spin (op, obra, oket)
        t1, w1 = logger.process_clock (), logger.perf_counter ()
        if len (set (inv)) == 4:
            self.dt_4fo += t1-t0
            self.dw_4fo += w1-w0
        else:
            use_gpu = getattr (param, 'use_gpu', False)
            if use_gpu: 
                self._gpu_opuniq_x_(op, obra, oket, ovecs, *inv)
                return 
        key = tuple ((obra, oket)) + inv
        inv = list (set (inv))
        brakets, bras, braHs = self.get_nonuniq_exc_square (key)
        for bra in bras:
            vec = ovecs[self.ox_ovlp_urootstr (bra, oket, inv)]
            self.put_ox1_(op.dot (vec.T).ravel (), bra, *inv)
            self._profile_4frag_(op)
        if len (braHs):
            t0, w0 = logger.process_clock (), logger.perf_counter ()
            op = op.conj ().T
            t1, w1 = logger.process_clock (), logger.perf_counter ()
            if len (set (inv)) == 4:
                self.dt_4fo += t1-t0
                self.dw_4fo += w1-w0
            for bra in braHs:
                vec = ovecs[self.ox_ovlp_urootstr (bra, obra, inv)]
                self.put_ox1_(op.dot (vec.T).ravel (), bra, *inv)
                self._profile_4frag_(op)
        return
    
    def _gpu_opuniq_x_ (self, op, obra, oket, ovecs, *inv):
        key = tuple ((obra, oket)) + inv
        inv = list (set (inv))
        brakets, bras, braHs = self.get_nonuniq_exc_square (key)

        self.gpu_matvec(op, bras, oket, ovecs, inv)

        if len (braHs):
            op = op.conj ().T
            self.gpu_matvec(op, braHs, obra, ovecs, inv)
        return

    def gpu_matvec(self, op, bras, oci, ovecs, inv):
        '''Effectively a op*vecs 
              where op is a matrix of size of m, k and 
              vecs several small vectors each of size n_i * k. 
              total_n = \sum_i n_i
           First run through ovecs collecting vecs in an array of size (total_n * k)
           Multiply them in a way that is optimal for gpu code 
           Store the result in a size of (m * total_n), and then do _put_ox1_ accordingly'''

        from mrh.my_pyscf.gpu import libgpu
        from pyscf.lib import param
        gpu = param.use_gpu
        #Get m,n,k for gemm gemm
        m, k = op.shape #m,k gemm
        libgpu.push_op(gpu, np.ascontiguousarray(op), m, k) #inits and pushes on all devices
        #Get n for gemm
        spec = np.ones (self.nfrags, dtype=bool)
        for i in inv: spec[i] = False
        spec = np.where (spec)[0]
        total_n = sum(np.prod (self.lroots[spec,bra]) for bra in bras)  #n

        libgpu.init_new_sivecs_host(gpu, m, total_n) #pinned memory for results

        libgpu.init_old_sivecs_host(gpu, k, total_n) #pinned memory for input

        n_array = []
        n_loc = 0
        for bra in bras:
            vec = ovecs[self.ox_ovlp_urootstr (bra, oci, inv)]
            n,_ = vec.shape #n,k gemm
            n_array.append(n)
            libgpu.push_sivecs_to_host(gpu, np.ascontiguousarray(vec), n_loc, n, k) #to pinned
            n_loc += n

        libgpu.compute_sivecs(gpu, m, total_n, k) # H2D, compute, D2H (pinned) 


        #vecs = []
        #for n in n_array:
        #    vecs.append(np.empty((n*m)))
        #n_loc = 0
        #for n, bra, vec in zip(n_array, bras, vecs):
        #    libgpu.pull_sivecs_from_pinned(gpu, vec, n_loc, m, n)
        #    self.put_ox1_(new_vec, bra, *inv)
        #    n_loc += n
        #vecs=None

        n_loc = 0
        for n, bra in zip(n_array, bras):
            vec = np.empty((m*n))
            libgpu.pull_sivecs_from_pinned(gpu, vec, n_loc, m, n)
            self.put_ox1_(vec, bra, *inv)
            n_loc += n

        return
    
    def ox_ovlp_urootstr (self, bra, ket, inv):
        '''Find the urootstr corresponding to the action of the overlap part of an operator
        from ket to bra, which might or might not be a part of the model space.'''
        urootstr = self.urootstr[:,bra].copy ()
        inv = list (inv)
        urootstr[inv] = self.urootstr[inv,ket]
        return tuple (urootstr)

    def ox_ovlp_uniq_str (self, ovlplink, ifrag):
        '''Find the unique source and destination urootstrs for applying the ifrag'th fragment's
        overlap part to the interactions tabulated in ovlplink'''
        # TODO: put the graph of connections in ovlplink, instead of re-finding them each time.
        vecstr = self.urootstr[:,ovlplink[:,0]].T
        vecstr[:,:ifrag] = ovlplink[:,1:ifrag+1]
        ovecstr = vecstr.copy ()
        ovecstr[:,ifrag] = ovlplink[:,ifrag+1]
        vecstr = np.unique (np.append (vecstr, ovecstr, axis=1), axis=0).reshape (-1,2,self.nfrags)
        vecstr, ovecstr = vecstr.transpose (1,0,2)
        return ovecstr, vecstr

    def ox_ovlp_frag (self, ovlplink, vecs, ifrag):
        '''Apply the ifrag'th fragment's overlap part of the interactions tabulated in ovlplink
        to the vectors collected in vecs'''
        ovecstr, vecstr = self.ox_ovlp_uniq_str (ovlplink, ifrag)
        ovecs = {tuple(os): 0 for os in np.unique (ovecstr, axis=0)}
        for os, s in zip (ovecstr, vecstr):
            vec = vecs[tuple(s)]
            lr = vec.shape[-1]
            bra, ket = os[ifrag], s[ifrag]
            o = self.ints[ifrag].get_ovlp (bra,ket,uroot_idx=True)
            lket = o.shape[1]
            vec = vec.reshape (-1,lket,lr)
            ovecs[tuple(os)] += lib.einsum ('pq,lqr->plr', o, vec).reshape (-1,lr)
        return ovecs

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
        self.log.timer ('HamS2OvlpOperators._ovlp_op', *t0)
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
        self.log.timer ('HamS2OvlpOperators.get_hdiag', *t0)
        return self.ox.copy ()

    def _crunch_hdiag_env_(self, _crunch_fn, *row): 
        if row[0] != row[1]: return
        if self._fn_row_has_spin (_crunch_fn):
            inv = row[2:-1]     
        else:
            inv = row[2:]
        ham, s2, sinv = _crunch_fn (*row)
        sinv = self.inv_unique (sinv)[::-1]
        key = tuple ((row[0], row[1])) + inv
        for braket in self.spman[key]:
            for bra, ket in self.nonuniq_exc[braket+inv]:
                if bra != ket: continue
                ham1 = opterm.reduce_spin (ham, bra, ket)
                hdiag_nonspec = self.get_hdiag_nonspectator (ham1, bra, *sinv)
                hdiag_spec = self.hdiag_spectator_ovlp (bra, *sinv)
                hdiag = np.multiply.outer (hdiag_nonspec, hdiag_spec)
                hdiag = transpose_sivec_with_slow_fragments (hdiag.ravel (), self.lroots[:,bra], *sinv)
                i, j = self.offs_lroots[bra] 
                self.ox[i:j] += hdiag.ravel ()

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

    def _crunch_2c_(self, bra, ket, a, i, b, j, s2lt, dry_run=False):
        '''Compute the reduced density matrix elements of a two-electron hop; i.e.,

        <bra|a'(s1)b'(s2)j(s2)i(s1)|ket>

        i.e.,

        i ---s1---> a
        j ---s2---> b

        with

        s2lt = 0, 1, 2
        s1   = a, a, b
        s2   = a, b, b

        and conjugate transpose

        Note that this includes a=b and/or i=j cases, but no other coincident fragment indices. Any
        other coincident fragment index (that is, any coincident index between the bra and the ket)
        turns this into one of the other interactions implemented in the above _crunch_ functions:
        s1 = s2  AND SORT (ab) = SORT (ij)                 : _crunch_1d_ and _crunch_2d_
        s1 = s2  AND (a = i XOR a = j XOR i = b XOR j = b) : _crunch_1c_ and _crunch_1c1d_
        s1 != s2 AND (a = j AND i = b)                     : _crunch_1s_
        s1 != s2 AND (a = j XOR i = b)                     : _crunch_1s1c_
        '''
        if (a==b) or (i==j): return super()._crunch_2c_(bra, ket, a, i, b, j, s2lt)
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        # s2lt: 0, 1, 2 -> aa, ab, bb
        # s2: 0, 1, 2, 3 -> aa, ab, ba, bb
        s2  = (0, 1, 3)[s2lt] # aa, ab, bb
        s2T = (0, 2, 3)[s2lt] # aa, ba, bb -> when you populate the e1 <-> e2 permutataon
        s11 = s2 // 2
        s12 = s2 % 2
        nelec_f_bra = self.nelec_rf[bra]
        nelec_f_ket = self.nelec_rf[ket]
        fac = (1,.5)[int ((a,i,s11)==(b,j,s12))] # 1/2 factor of h2 canceled by aibj <-> bjai
        fac *= (1,-1)[int (a>b)]
        fac *= fermion_des_shuffle (nelec_f_bra, (a, i, b, j), a)
        fac *= fermion_des_shuffle (nelec_f_bra, (a, i, b, j), b)
        fac *= (1,-1)[int (i>j)]
        fac *= fermion_des_shuffle (nelec_f_ket, (a, i, b, j), i)
        fac *= fermion_des_shuffle (nelec_f_ket, (a, i, b, j), j)
        ham = self.get_ham_2q (j,b,i,a).transpose (0,2,3,1) # BEWARE CONJ
        if s11==s12: # exchange
            ham -= self.get_ham_2q (j,a,i,b).transpose (0,2,1,3) # BEWARE CONJ
        ham *= fac
        d_b = self.ints[b].get_p (bra, ket, s12, highm=True)
        d_a = self.ints[a].get_p (bra, ket, s11, highm=True)
        d_i = self.ints[i].get_h (bra, ket, s11, highm=True)
        d_j = self.ints[j].get_h (bra, ket, s12, highm=True)
        frags = (j,i,a,b)
        d = (d_j,d_i,d_a,d_b)
        ints = [self.ints[p] for p in frags]
        ham = opterm.OpTerm4Fragments (ham, frags, d, ints, do_crunch=(not dry_run))
        s2 = None
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_2c, self.dw_2c = self.dt_2c + dt, self.dw_2c + dw
        return ham, s2, (j, i, a, b)


#gen_contract_op_si_hdiag = functools.partial (_fake_gen_contract_op_si_hdiag, ham)
def gen_contract_op_si_hdiag (las, h1, h2, ci, nelec_frs, smult_fr=None, soc=0, nlas=None,
                              _HamS2Ovlp_class=HamS2OvlpOperators, _return_int=False,
                              screen_thresh=SCREEN_THRESH, **kwargs):
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
        smult_fr : ndarray of shape (nfrags,nroots)
            Spin multiplicity of each fragment in each rootspace
        soc : integer
            Order of spin-orbit coupling included in the Hamiltonian
        nlas : sequence of length (nfrags)
            Number of orbitals in each fragment
        _HamS2Ovlp_class : class
            The main intermediate class
        _return_int : logical
            If True, return the main intermediate object instead of the
            operator matrices
        screen_thresh : float
            Tolerance for screening Hamiltonian and S^2 operator components
        
    Returns: 
        ham_op : LinearOperator of shape (nstates,nstates)
            Hamiltonian in LAS product state basis
        s2_op : LinearOperator of shape (nstates,nstates)
            Spin-squared operator in LAS product state basis
        ovlp_op : LinearOperator of shape (nstates,nstates)
            Overlap matrix of LAS product states 
        hdiag : ndarray of shape (nstates,)
            Diagonal element of Hamiltonian matrix
        _get_ovlp : callable with kwarg rootidx
            Produce the overlap matrix between model states in a set of rootspaces,
            identified by ndarray or list "rootidx"
    '''
    t1 = (lib.logger.process_clock (), lib.logger.perf_counter ())
    verbose = kwargs.get ('verbose', las.verbose)
    log = lib.logger.new_logger (las, verbose)
    if nlas is None: nlas = las.ncas_sub
    pt_order = kwargs.get ('pt_order', None)
    do_pt_order = kwargs.get ('do_pt_order', None)
    max_memory = getattr (las, 'max_memory', las.mol.max_memory)
    dtype = h1.dtype
    nfrags, nroots = nelec_frs.shape[:2]
    if soc>1: raise NotImplementedError ("Spin-orbit coupling of second order")

    t1 = log.timer ('LASSI hsi operator setup', *t1)
    # Handle possible SOC
    spin_pure, h1, h2, ci, nelec_frs, smult_fr, nlas, spin_shuffle_fac = soc_context (
        h1, h2, ci, nelec_frs, smult_fr, soc, nlas)
    t1 = log.timer ('LASSI hsi operator soc handling', *t1)

    # First pass: single-fragment intermediates
    ints, lroots = frag.make_ints (las, ci, nelec_frs, nlas=nlas, smult_fr=smult_fr,
                                   pt_order=pt_order, do_pt_order=do_pt_order, verbose=verbose)
    t1 = log.timer ('LASSI hsi operator first pass make ints', *t1)
    nstates = np.sum (np.prod (lroots, axis=0))

    t1 = log.timer ('LASSI hsi operator first pass nstates making', *t1)
    # Second pass: upper-triangle
    t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
    outerprod = _HamS2Ovlp_class (ints, nlas, lroots, h1, h2,
                                  pt_order=pt_order, do_pt_order=do_pt_order,
                                  dtype=dtype, max_memory=max_memory, log=log,
                                  screen_thresh=screen_thresh)

    t1 = log.timer ('LASSI hsi operator hams2ovlp class', *t1)
    if soc and not spin_pure:
        outerprod.spin_shuffle = spin_shuffle_fac
    t1 = log.timer ('LASSI hsi operator spin shuffle assigning?', *t1)
    log.timer ('LASSI hsi operator build', *t0)

    if _return_int: return outerprod

    ham_op = outerprod.get_ham_op ()
    s2_op = outerprod.get_s2_op ()
    ovlp_op = outerprod.get_ovlp_op ()
    hdiag = outerprod.get_hdiag ()
    #raw2orth = citools.get_orth_basis (ci, las.ncas_sub, nelec_frs,
    #                                   _get_ovlp=outerprod.get_ovlp)
    return ham_op, s2_op, ovlp_op, hdiag, outerprod.get_ovlp

