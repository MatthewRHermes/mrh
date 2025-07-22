import numpy as np
from scipy import linalg
from pyscf import lib
from pyscf.fci.direct_spin1 import trans_rdm12s, trans_rdm1s
from pyscf.fci.direct_spin1 import contract_1e, contract_2e, absorb_h1e
from pyscf.fci.direct_uhf import contract_1e as contract_1e_uhf
from pyscf.fci.addons import cre_a, cre_b, des_a, des_b
from pyscf.fci import cistring
from itertools import product, combinations
from mrh.my_pyscf.lassi.citools import get_lroots, get_rootaddr_fragaddr, get_unique_roots
from mrh.my_pyscf.lassi.op_o1.utilities import *
from mrh.my_pyscf.fci.rdm import trans_rdm1ha_des, trans_rdm1hb_des
from mrh.my_pyscf.fci.rdm import trans_rdm13ha_des, trans_rdm13hb_des
from mrh.my_pyscf.fci.rdm import trans_sfddm1, trans_hhdm
from mrh.my_pyscf.fci.direct_halfelectron import contract_1he, absorb_h1he, contract_3he
from mrh.my_pyscf.fci.direct_nosym_uhf import contract_1e as contract_1e_nosym_uhf
from mrh.my_pyscf.fci.direct_nosym_ghf import contract_1e as contract_1e_nosym_ghf
from mrh.my_pyscf.fci.pair_op import contract_pair_op
from pyscf import __config__
import functools
import copy

SCREEN_THRESH = getattr (__config__, 'lassi_frag_screen_thresh', 1e-10)
DO_SCREEN_LINEQUIV = getattr (__config__, 'lassi_frag_do_screen_linequiv', True)

class FragTDMInt (object):
    ''' Fragment-local LAS state transition density matrix intermediate

        Quasi-sparse-memory storage for LAS-state transition density matrix single-fragment
        intermediates. Stores all local transition density matrix factors. Run the `kernel` method
        to compute all data, and run the get_* methods to return the computed intermediates:

        s and t are spin: a,b for 1 operator; aa, ab, bb for 2 operators
        s is a spin argument passed to the "get" function
        t is a spin index on the returned array
        i and j are single state indices with
            rootaddr[i] = index of the rootspace for state i
            fragaddr[i] = index in this fragment's local rootaddr[i] basis of state i

        get_h (i,j,s): <i|s|j>
        get_p (i,j,s): <i|s'|j> = conj (<j|s|i>)
        get_dm1 (i,j): <i|t't|j>
        get_hh (i,j,s): <i|s2s1|j>
        get_pp (i,j,s): <i|s1's2'|j> = conj (<j|s2s1|i>)
        get_sm (i,j): <i|b'a|j>
        get_sp (i,j): <i|a'b|j> = conj (<j|b'a|i>)
        get_phh (i,j,s): <i|t'ts|j>
        get_pph (i,j,s): <i|s't't|j> = conj (<j|t'ts|i>)
        get_dm2 (i,j): <i|t1't2't2t1|j>

        TODO: two-electron spin-broken components
            <i|a'b'bb|j> & h.c. & a<->b
            <i|a'a'bb|j> & a<->b
        Req'd for 2e- relativistic (i.e., spin-breaking) operators

        NOTE: in the set_* and get_* functions, the indices i,j are rootspace indices and the major
        axis is the lroot axis. In the get_1_* functions, on the other hand, the indices i,j are
        single model state indices.

        Args:
            las : instance of :class:`LASCINoSymm`
                Only las.stdout and las.verbose (sometimes) are used to direct the logger output
            ci : list of ndarray of length nroots
                Contains CI vectors for the current fragment
            hopping_index: ndarray of ints of shape (2, nroots, nroots)
                element [i,j,k] reports the change of number of electrons of
                spin i in the current fragment between LAS rootspaces j and k
            zerop_index : ndarray of bools of shape (nroots, nroots)
                element [i,j] is true where the ith and jth LAS spaces are
                connected by a null excitation; i.e., no electron, pair,
                or spin hopping or pair splitting/coalescence. This implies
                nonzero 1- and 2-body transition density matrices within
                all fragments.
            onep_index : ndarray of bools of shape (nroots, nroots)
                element [i,j] is true where the ith and jth LAS spaces
                are connected by exactly one electron hop from i to j or vice
                versa, implying nonzero 1-body transition density matrices
                within spectator fragments and phh/pph modes within
                source/dest fragments.
            norb : integer
                number of active orbitals in the current fragment
            nroots : integer
                number of states considered
            nelec_rs : ndarray of ints of shape (nroots, 2)
                number of spin-up and spin-down electrons in each root
            rootaddr: ndarray of shape (nstates):
                Index array of LAS states into a given rootspace
            fragaddr: ndarray of shape (nstates):
                Index array of LAS states into this fragment's local basis
            idx_frag : integer
                index label of current fragment

        Kwargs:
            dtype : instance of np.dtype
                Currently not used
            discriminator : sequence of length nroots
                Additional information to discriminate between otherwise-equivalent rootspaces
            screen_linequiv : logical
                Whether to compress data by aggressively identifying linearly equivalent
                rootspaces and storing the relevant unitary matrices.
            verbose : integer
                Logger verbosity level
    '''

    def __init__(self, las, ci, hopping_index, zerop_index, onep_index, norb, nroots, nelec_rs,
                 rootaddr, fragaddr, idx_frag, mask_ints, dtype=np.float64, discriminator=None,
                 pt_order=None, do_pt_order=None, screen_linequiv=DO_SCREEN_LINEQUIV,
                 verbose=None):
        # TODO: if it actually helps, cache the "linkstr" arrays
        if verbose is None: verbose = las.verbose
        self.verbose = verbose
        self.log = lib.logger.new_logger (las, self.verbose)
        self.ci = ci
        self.hopping_index = hopping_index
        self.zerop_index = zerop_index
        self.onep_index = onep_index
        self.norb = norb
        self.nroots = nroots
        self.dtype = dtype
        self.nelec_r = [tuple (n) for n in nelec_rs]
        self.linkstr_cache = {}
        self.linkstrl_cache = {}
        self.rootaddr = rootaddr
        self.rootinvaddr = np.unique (rootaddr, return_index=True)[1]
        self.fragaddr = fragaddr
        self.idx_frag = idx_frag
        self.mask_ints = mask_ints
        self.discriminator = discriminator

        if pt_order is None: pt_order = np.zeros (nroots, dtype=int)
        self.pt_order = pt_order
        self.do_pt_order = do_pt_order

        # Consistent array shape
        self.ndeta_r = np.array ([cistring.num_strings (norb, nelec[0]) for nelec in self.nelec_r])
        self.ndetb_r = np.array ([cistring.num_strings (norb, nelec[1]) for nelec in self.nelec_r])
        self.ci = [c.reshape (-1,na,nb) for c, na, nb in zip (self.ci, self.ndeta_r, self.ndetb_r)]

        self.time_crunch = self._init_crunch_(screen_linequiv)

    def _check_linkstr_cache (self, no, na, nb):
        if (no, na, nb) not in self.linkstr_cache.keys ():
            la = cistring.gen_linkstr_index(range(no), na)
            lb = cistring.gen_linkstr_index(range(no), nb)
            linkstr = (la,lb)
            self.linkstr_cache[(no,na,nb)] = (la,lb)
            return linkstr
        else:
            return self.linkstr_cache[(no,na,nb)]

    def _check_linkstrl_cache (self, no, na, nb):
        if (no, na, nb) not in self.linkstrl_cache.keys ():
            la = cistring.gen_linkstr_index_trilidx(range(no), na)
            lb = cistring.gen_linkstr_index_trilidx(range(no), nb)
            linkstrl = (la,lb)
            self.linkstrl_cache[(no,na,nb)] = (la,lb)
            return linkstrl
        else:
            return self.linkstrl_cache[(no,na,nb)]

    # Exception catching

    def try_get_1 (self, tab, *args):
        i, j = args[-2:]
        ir, jr = self.rootaddr[i], self.rootaddr[j]
        ip, jp = self.fragaddr[i], self.fragaddr[j]
        rargs = [x for x in args[:-2]] + [ir,jr]
        return self.try_get (tab, *rargs)[ip,jp]

    def try_get (self, tab, *args):
        if len (args) == 3: return self.try_get_tdm (tab, *args)
        elif len (args) == 2: return self.try_get_dm (tab, *args)
        else: raise RuntimeError (str (len (args)))

    def try_get_dm (self, tab, i, j):
        ir, jr = self.uroot_idx[i], self.uroot_idx[j]
        try:
            assert (tab[ir][jr] is not None)
            return tab[ir][jr]
        except Exception as e:
            errstr = 'frag {} failure to get element {},{}'.format (self.idx_frag, ir, jr)
            errstr = errstr + '\nhopping_index entry: {}'.format (self.hopping_index[:,ir,jr])
            raise RuntimeError (errstr)

    def try_get_tdm (self, tab, s, i, j):
        ir, jr = self.uroot_idx[i], self.uroot_idx[j]
        try:
            assert (tab[s][ir][jr] is not None)
            return tab[s][ir][jr]
        except Exception as e:
            errstr = 'frag {} failure to get element {},{} w spin {}'.format (
                self.idx_frag, ir, jr, s)
            errstr = errstr + '\nhopping_index entry: {}'.format (self.hopping_index[:,ir,jr])
            raise RuntimeError (errstr)

    def setmanip (self, x): return np.ascontiguousarray (x)

    # 0-particle intermediate (overlap)

    def get_ovlp (self, i, j):
        return self.try_get (self.ovlp, i, j)

    def get_ovlp_inpbasis (self, i, j):
        ''' Apply umat if present to get the actual original-basis overlap '''
        ovlp = self.get_ovlp (i, j)
        if i in self.umat_root:
            ovlp = np.dot (self.umat_root[i].conj ().T, ovlp)
        if j in self.umat_root:
            ovlp = np.dot (ovlp, self.umat_root[j])
        return ovlp

    def get_1_ovlp (self, i, j):
        return self.try_get_1 (self.ovlp, i, j)

    # 1-particle 1-operator intermediate

    def get_h (self, i, j, s):
        return self.try_get (self._h, s, i, j)

    def set_h (self, i, j, s, x):
        i, j = self.uroot_idx[i], self.uroot_idx[j]
        x = self.setmanip (x)
        self._h[s][i][j] = x
        return x

    def get_p (self, i, j, s):
        return self.try_get (self._h, s, j, i).conj ().transpose (1,0,2)

    def get_1_h (self, i, j, s):
        return self.try_get_1 (self._h, s, i, j)

    def get_1_p (self, i, j, s):
        return self.try_get_1 (self._h, s, j, i).conj ()

    # 2-particle intermediate

    def get_hh (self, i, j, s):
        return self.try_get (self._hh, s, i, j)
        #return self._hh[s][i][j]

    def set_hh (self, i, j, s, x):
        i, j = self.uroot_idx[i], self.uroot_idx[j]
        x = self.setmanip (x)
        self._hh[s][i][j] = x
        return x

    def get_pp (self, i, j, s):
        return self.try_get (self._hh, s, j, i).conj ().transpose (1,0,3,2)

    def get_1_hh (self, i, j, s):
        return self.try_get_1 (self._hh, s, i, j)
        #return self._hh[s][i][j]

    def get_1_pp (self, i, j, s):
        return self.try_get_1 (self._hh, s, j, i).conj ().T

    # 1-particle 3-operator intermediate

    def get_phh (self, i, j, s):
        return self.try_get (self._phh, s, i, j)

    def set_phh (self, i, j, s, x):
        i, j = self.uroot_idx[i], self.uroot_idx[j]
        x = self.setmanip (x)
        self._phh[s][i][j] = x
        return x

    def get_pph (self, i, j, s):
        return self.try_get (self._phh, s, j, i).conj ().transpose (1,0,2,5,4,3)

    def get_1_phh (self, i, j, s):
        return self.try_get_1 (self._phh, s, i, j)

    def get_1_pph (self, i, j, s):
        return self.try_get_1 (self._phh, s, j, i).conj ().transpose (0,3,2,1)

    # spin-hop intermediate

    def get_sm (self, i, j):
        return self.try_get (self._sm, i, j)

    def set_sm (self, i, j, x):
        i, j = self.uroot_idx[i], self.uroot_idx[j]
        x = self.setmanip (x)
        self._sm[i][j] = x
        return x

    def get_sp (self, i, j):
        return self.try_get (self._sm, j, i).conj ().transpose (1,0,3,2)

    def get_1_sm (self, i, j):
        return self.try_get_1 (self._sm, i, j)

    def get_1_sp (self, i, j):
        return self.try_get_1 (self._sm, j, i).conj ().T

    def get_smp (self, i, j, s):
        if s==0: return self.get_sm (i, j)
        elif s==1: return self.get_sp (i, j)
        else: raise RuntimeError

    def get_1_smp (self, i, j, s):
        if s==0: return self.get_1_sm (i, j)
        elif s==1: return self.get_1_sp (i, j)
        else: raise RuntimeError

    # 1-density intermediate

    def get_dm1 (self, i, j):
        if self.unique_root[j] > self.unique_root[i]:
            return self.try_get (self.dm1, j, i).conj ().transpose (1,0,2,4,3)
        return self.try_get (self.dm1, i, j)

    def set_dm1 (self, i, j, x):
        assert (j <= i)
        i, j = self.uroot_idx[i], self.uroot_idx[j]
        x = self.setmanip (x)
        self.dm1[i][j] = x

    def get_1_dm1 (self, i, j):
        if self.unique_root[self.rootaddr[j]] > self.unique_root[self.rootaddr[i]]:
            return self.try_get_1 (self.dm1, j, i).conj ().transpose (0, 2, 1)
        return self.try_get_1 (self.dm1, i, j)

    # 2-density intermediate

    def get_dm2 (self, i, j):
        if self.unique_root[j] > self.unique_root[i]:
            return self.try_get (self.dm2, j, i).conj ().transpose (1,0,2,4,3,6,5)
        return self.try_get (self.dm2, i, j)

    def get_1_dm2 (self, i, j):
        if self.unique_root[self.rootaddr[j]] > self.unique_root[self.rootaddr[i]]:
            return self.try_get_1 (self.dm2, j, i).conj ().transpose (0, 2, 1, 4, 3)
        return self.try_get_1 (self.dm2, i, j)

    def set_dm2 (self, i, j, x):
        assert (j <= i)
        i, j = self.uroot_idx[i], self.uroot_idx[j]
        x = self.setmanip (x)
        self.dm2[i][j] = x

    def get_lroots (self, i):
        return self.ci[i].shape[0]

    def get_lroots_uroot (self, i):
        return self.get_lroots (self.uroot_addr[i])

    def _init_crunch_(self, screen_linequiv):
        ''' Compute the transition density matrix factors.

        Returns:
            t0 : tuple of length 2
                timestamp of entry into this function, for profiling by caller
        '''
        ci = self.ci
        ndeta, ndetb = self.ndeta_r, self.ndetb_r
        hopping_index = self.hopping_index
        zerop_index = self.zerop_index
        onep_index = self.onep_index
        self.mask_ints = np.logical_or (
            self.mask_ints, self.mask_ints.T
        )
        if self.do_pt_order is not None:
            pt_mask = np.add.outer (self.pt_order, self.pt_order)
            pt_mask = np.isin (pt_mask, self.do_pt_order)
            self.mask_ints = np.logical_and (
                self.mask_ints, pt_mask
            )
                
        self.root_unique, self.unique_root, self.umat_root = get_unique_roots (
            ci, self.nelec_r, screen_linequiv=screen_linequiv, screen_thresh=SCREEN_THRESH,
            discriminator=self.discriminator
        )
        self.nuroots = nuroots = np.count_nonzero (self.root_unique)
        self.uroot_inv = -1 * np.ones (self.nroots, dtype=int)
        self.uroot_inv[self.root_unique] = np.arange (nuroots, dtype=int)
        self.uroot_idx = self.uroot_inv[self.unique_root]
        self.uroot_addr = np.where (self.root_unique)[0]
        assert (np.all (self.uroot_idx >= 0))

        self.ovlp = [[None for i in range (nuroots)] for j in range (nuroots)]
        self._h = [[[None for i in range (nuroots)] for j in range (nuroots)] for s in (0,1)]
        self._hh = [[[None for i in range (nuroots)] for j in range (nuroots)] for s in (-1,0,1)] 
        self._phh = [[[None for i in range (nuroots)] for j in range (nuroots)] for s in (0,1)]
        self._sm = [[None for i in range (nuroots)] for j in range (nuroots)]
        self.dm1 = [[None for i in range (nuroots)] for j in range (nuroots)]
        self.dm2 = [[None for i in range (nuroots)] for j in range (nuroots)]

        # Update connectivity arrays and mask_ints
        for i in np.where (self.root_unique)[0]:
            images = np.where (self.unique_root==i)[0]
            for j in images:
                self.onep_index[i] |= self.onep_index[j]
                self.onep_index[:,i] |= self.onep_index[:,j]
                self.zerop_index[i] |= self.zerop_index[j]
                self.zerop_index[:,i] |= self.zerop_index[:,j]
                self.mask_ints[i,:] = np.logical_or (
                    self.mask_ints[i,:], self.mask_ints[j,:]
                )
                self.mask_ints[:,i] = np.logical_or (
                    self.mask_ints[:,i], self.mask_ints[:,j]
                )

        return self._make_dms_()

    def update_ci_(self, iroot, ci):
        update_mask = np.zeros ((self.nroots, self.nroots), dtype=bool)
        for i, civec in zip (iroot, ci):
            assert (self.root_unique[i]), 'Cannot update non-unique CI vectors'
            update_mask[i,:] = True
            update_mask[:,i] = True
            self.ci[i] = civec.reshape (-1, self.ndeta_r[i], self.ndetb_r[i])
        mask_ints = self.mask_ints & update_mask
        t0 = self._make_dms_(mask_ints=mask_ints)
        self.log.timer ('Update density matrices of fragment intermediate', *t0)

    def _make_dms_(self, mask_ints=None):
        if mask_ints is None: mask_ints=self.mask_ints
        t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
        ci = self.ci
        ndeta, ndetb = self.ndeta_r, self.ndetb_r
        hopping_index = self.hopping_index
        zerop_index = self.zerop_index
        onep_index = self.onep_index
        idx_uniq = self.root_unique
        lroots = [c.shape[0] for c in ci]
        nroots, norb = self.nroots, self.norb

        # Overlap matrix
        offs = np.cumsum (lroots)
        for i, j in combinations (np.where (idx_uniq)[0], 2):
            if self.nelec_r[i] != self.nelec_r[j]: continue
            if not mask_ints[i,j]: continue
            ci_i = ci[i].reshape (lroots[i], -1)
            ci_j = ci[j].reshape (lroots[j], -1)
            k, l = self.uroot_idx[i], self.uroot_idx[j]
            self.ovlp[k][l] = np.dot (ci_i.conj (), ci_j.T)
            self.ovlp[l][k] = self.ovlp[k][l].conj ().T
        for i in np.where (idx_uniq)[0]:
            if not mask_ints[i,i]: continue
            ci_i = ci[i].reshape (lroots[i], -1)
            j = self.uroot_idx[i]
            self.ovlp[j][j] = np.dot (ci_i.conj (), ci_i.T)
            #errmat = self.ovlp[i][i] - np.eye (lroots[i])
            #if np.amax (np.abs (errmat)) > 1e-3:
            #    w, v = np.linalg.eigh (self.ovlp[i][i])
            #    errmsg = ('States w/in single Hilbert space must be orthonormal; '
            #              'eigvals (ovlp) = {}')
            #    raise RuntimeError (errmsg.format (w))

        # Loop over lroots functions
        def des_loop (des_fn, c, nelec, p):
            #na = cistring.num_strings (norb, nelec[0])
            #nb = cistring.num_strings (norb, nelec[1])
            #c = c.reshape (-1, na, nb)
            des_c = [des_fn (c_i, norb, nelec, p) for c_i in c]
            assert (c.ndim==3)
            return np.asarray (des_c)
        def des_a_loop (c, nelec, p): return des_loop (des_a, c, nelec, p)
        def des_b_loop (c, nelec, p): return des_loop (des_b, c, nelec, p)
        def trans_rdm12s_loop (iroot, bra, ket, do2=True):
            nelec = self.nelec_r[iroot]
            na, nb = ndeta[iroot], ndetb[iroot]
            bra = bra.reshape (-1, na, nb)
            ket = ket.reshape (-1, na, nb)
            tdm1s = np.zeros ((bra.shape[0],ket.shape[0],2,norb,norb), dtype=self.dtype)
            tdm2s = np.zeros ((bra.shape[0],ket.shape[0],4,norb,norb,norb,norb), dtype=self.dtype)
            linkstr = self._check_linkstr_cache (norb, nelec[0], nelec[1])
            if do2:
                for i, j in product (range (bra.shape[0]), range (ket.shape[0])):
                    d1s, d2s = trans_rdm12s (bra[i], ket[j], norb, nelec,
                                             link_index=linkstr)
                    # Transpose based on docstring of direct_spin1.trans_rdm12s
                    tdm1s[i,j] = np.stack (d1s, axis=0).transpose (0, 2, 1)
                    tdm2s[i,j] = np.stack (d2s, axis=0)
            else:
                for i, j in product (range (bra.shape[0]), range (ket.shape[0])):
                    d1s = trans_rdm1s (bra[i], ket[j], norb, nelec,
                                       link_index=linkstr)
                    # Transpose based on docstring of direct_spin1.trans_rdm12s
                    tdm1s[i,j] = np.stack (d1s, axis=0).transpose (0, 2, 1)
            return tdm1s, tdm2s
        def trans_rdm1s_loop (iroot, bra, ket):
            return trans_rdm12s_loop (iroot, bra, ket, do2=False)[0]
        def trans_rdm13h_loop (bra_r, ket_r, spin=0, do3h=True):
            trans_rdm13h = (trans_rdm13ha_des, trans_rdm13hb_des)[spin]
            trans_rdm1h = (trans_rdm1ha_des, trans_rdm1hb_des)[spin]
            nelec_ket = self.nelec_r[ket_r]
            bravecs = ci[bra_r].reshape (-1, ndeta[bra_r], ndetb[bra_r])
            ketvecs = ci[ket_r].reshape (-1, ndeta[ket_r], ndetb[ket_r])
            tdm1h = np.zeros ((bravecs.shape[0],ketvecs.shape[0],norb), dtype=self.dtype)
            tdm3h = np.zeros ((bravecs.shape[0],ketvecs.shape[0],2,norb,norb,norb),
                              dtype=self.dtype)
            linkstr = self._check_linkstr_cache (norb+1, nelec_ket[0], nelec_ket[1])
            if do3h:
                for i, j in product (range (bravecs.shape[0]), range (ketvecs.shape[0])):
                    d1s, d2s = trans_rdm13h (bravecs[i], ketvecs[j], norb, nelec_ket,
                                             link_index=linkstr)
                    tdm1h[i,j] = d1s
                    tdm3h[i,j] = np.stack (d2s, axis=0).transpose (0,2,3,1)
                    # Mulliken -> Dirac
            else:
                for i, j in product (range (bravecs.shape[0]), range (ketvecs.shape[0])):
                    d1s = trans_rdm1h (bravecs[i], ketvecs[j], norb, nelec_ket,
                                       link_index=linkstr)
                    tdm1h[i,j] = d1s
            return tdm1h, tdm3h
        def trans_rdm1h_loop (bra_r, ket_r, spin=0):
            return trans_rdm13h_loop (bra_r, ket_r, do3h=False)[0]
        def trans_sfddm_loop (bra_r, ket_r):
            bravecs = ci[bra_r].reshape (-1, ndeta[bra_r], ndetb[bra_r])
            ketvecs = ci[ket_r].reshape (-1, ndeta[ket_r], ndetb[ket_r])
            nelec_ket = self.nelec_r[ket_r]
            sfddm = np.zeros ((bravecs.shape[0],ketvecs.shape[0],norb,norb), dtype=self.dtype)
            linkstr = self._check_linkstr_cache (norb+1, nelec_ket[0], nelec_ket[1]+1)
            for i, j in product (range (bravecs.shape[0]), range (ketvecs.shape[0])):
                d1 = trans_sfddm1 (bravecs[i], ketvecs[j], norb, nelec_ket,
                                   link_index=linkstr)
                sfddm[i,j] = d1
            return sfddm
        def trans_hhdm_loop (bra_r, ket_r, spin=0):
            bravecs = ci[bra_r].reshape (-1, ndeta[bra_r], ndetb[bra_r])
            ketvecs = ci[ket_r].reshape (-1, ndeta[ket_r], ndetb[ket_r])
            nelec_ket = self.nelec_r[ket_r]
            hhdm = np.zeros ((bravecs.shape[0],ketvecs.shape[0],norb,norb), dtype=self.dtype)
            ndum = 2 - (spin%2)
            linkstr = self._check_linkstr_cache (norb+ndum, nelec_ket[0], nelec_ket[1])
            for i, j in product (range (bravecs.shape[0]), range (ketvecs.shape[0])):
                d1 = trans_hhdm (bravecs[i], ketvecs[j], norb, nelec_ket,
                                 spin=spin, link_index=linkstr)
                hhdm[i,j] = d1
            return hhdm

        # Spectator fragment contribution
        spectator_index = np.all (hopping_index == 0, axis=0)
        spectator_index[~idx_uniq,:] = False
        spectator_index[:,~idx_uniq] = False
        spectator_index[np.triu_indices (nroots, k=1)] = False
        spectator_index = np.stack (np.where (spectator_index), axis=1)
        for i, j in spectator_index:
            if not mask_ints[i,j]: continue
            dm1s, dm2s = trans_rdm12s_loop (j, ci[i], ci[j], do2=zerop_index[i,j])
            self.set_dm1 (i, j, dm1s)
            if zerop_index[i,j]: self.set_dm2 (i, j, dm2s)
 
        hidx_ket_a = np.where (np.any (hopping_index[0] < 0, axis=0) & idx_uniq)[0]
        hidx_ket_b = np.where (np.any (hopping_index[1] < 0, axis=0) & idx_uniq)[0]

        # a_p|i>; shape = (norb, lroots[ket], ndeta[*], ndetb[ket])
        for ket in hidx_ket_a:
            for bra in np.where ((hopping_index[0,:,ket] < 0) & idx_uniq)[0]:
                if not mask_ints[bra,ket]: continue
                # <j|a_p|i>
                if np.all (hopping_index[:,bra,ket] == [-1,0]):
                    h, phh = trans_rdm13h_loop (bra, ket, spin=0)
                    self.set_h (bra, ket, 0, h)
                    # <j|a'_q a_r a_p|i>, <j|b'_q b_r a_p|i> - how to tell if consistent sign rule?
                    if onep_index[bra,ket]:
                        err = np.abs (phh[:,:,0] + phh[:,:,0].transpose (0,1,2,4,3))
                        assert (np.amax (err) < 1e-8), '{}'.format (np.amax (err)) 
                        # ^ Passing this assert proves that I have the correct index
                        # and argument ordering for the call and return of trans_rdm12s
                        self.set_phh (bra, ket, 0, phh)
                # <j|b'_q a_p|i> = <j|s-|i>
                elif np.all (hopping_index[:,bra,ket] == [-1,1]):
                    self.set_sm (bra, ket, trans_sfddm_loop (bra, ket))
                # <j|b_q a_p|i>
                elif np.all (hopping_index[:,bra,ket] == [-1,-1]):
                    self.set_hh (bra, ket, 1, trans_hhdm_loop (bra, ket, spin=1))
                # <j|a_q a_p|i>
                elif np.all (hopping_index[:,bra,ket] == [-2,0]):
                    self.set_hh (bra, ket, 0, trans_hhdm_loop (bra, ket, spin=0))
                
        # b_p|i>
        for ket in hidx_ket_b:
            for bra in np.where ((hopping_index[1,:,ket] < 0) & idx_uniq)[0]:
                if not mask_ints[bra,ket]: continue
                # <j|b_p|i>
                if np.all (hopping_index[:,bra,ket] == [0,-1]):
                    h, phh = trans_rdm13h_loop (bra, ket, spin=1)
                    self.set_h (bra, ket, 1, h)
                    # <j|a'_q a_r b_p|i>, <j|b'_q b_r b_p|i> - how to tell if consistent sign rule?
                    if onep_index[bra,ket]:
                        err = np.abs (phh[:,:,1] + phh[:,:,1].transpose (0,1,2,4,3))
                        assert (np.amax (err) < 1e-8), '{}'.format (np.amax (err))
                        # ^ Passing this assert proves that I have the correct index
                        # and argument ordering for the call and return of trans_rdm12s
                        self.set_phh (bra, ket, 1, phh)
                # <j|b_q b_p|i>
                elif np.all (hopping_index[:,bra,ket] == [0,-2]):
                    self.set_hh (bra, ket, 2, trans_hhdm_loop (bra, ket, spin=2))
        
        return t0

    def symmetrize_pt1_(self, ptmap):
        ''' Symmetrize transition density matrices of first order in perturbation theory '''
        pt_mask = np.add.outer (self.pt_order, self.pt_order)
        pt_mask = pt_mask==1
        pt_mask[:,self.pt_order==1] = False
        mask_ints = np.logical_and (
           self.mask_ints, pt_mask
        )
        ptmap = np.append (ptmap, ptmap[:,::-1], axis=0)
        ptmap = {i:j for i,j in ptmap}
        idx_uniq = self.root_unique
        for i, j in combinations (np.where (idx_uniq)[0], 2):
            if not mask_ints[i,j]: continue
            if self.nelec_r[i] != self.nelec_r[j]: continue
            k, l = ptmap[i], ptmap[j]
            o = (self.get_ovlp (i,j) + self.get_ovlp (k,l)) / 2
            self.ovlp[i][j] = self.ovlp[k][l] = o

        # Spectator fragment contribution
        hopping_index = self.hopping_index
        zerop_index = self.zerop_index
        onep_index = self.onep_index
        idx_uniq = self.root_unique
        nroots = self.nroots
        spectator_index = np.all (hopping_index == 0, axis=0)
        spectator_index[~idx_uniq,:] = False
        spectator_index[:,~idx_uniq] = False
        spectator_index = np.stack (np.where (spectator_index), axis=1)
        for i, j in spectator_index:
            if not mask_ints[i,j]: continue
            k, l = ptmap[i], ptmap[j]
            dm1s = (self.get_dm1 (i, j) + self.get_dm1 (k, l)) / 2
            self.set_dm1 (i, j, dm1s)
            if k >= l:
                self.set_dm1 (k, l, dm1s)
            else:
                self.set_dm1 (l, k, dm1s.conj ().transpose (1,0,2,4,3))
            if not zerop_index[i,j]: continue
            dm2s = (self.get_dm2 (i, j) + self.get_dm2 (k, l)) / 2
            self.set_dm2 (i, j, dm2s)
            if k < l: k, l, dm2s = l, k, dm2s.conj ().transpose (1,0,2,4,3,6,5)
            self.set_dm2 (k, l, dm2s)

        hidx_ket_a = np.where (np.any (hopping_index[0] < 0, axis=0) & idx_uniq)[0]
        hidx_ket_b = np.where (np.any (hopping_index[1] < 0, axis=0) & idx_uniq)[0]

        # a_p|i>; shape = (norb, lroots[ket], ndeta[*], ndetb[ket])
        for ket in hidx_ket_a:
            for bra in np.where ((hopping_index[0,:,ket] < 0) & idx_uniq)[0]:
                if not mask_ints[bra,ket]: continue
                bet = ptmap[bra]
                kra = ptmap[ket]
                # <j|a_p|i>
                if np.all (hopping_index[:,bra,ket] == [-1,0]):
                    h = (self.get_h (bra, ket, 0) + self.get_h (bet, kra, 0)) / 2
                    self.set_h (bra, ket, 0, h)
                    self.set_h (bet, kra, 0, h)
                    # <j|a'_q a_r a_p|i>, <j|b'_q b_r a_p|i> - how to tell if consistent sign rule?
                    if onep_index[bra,ket]:
                        phh = (self.get_phh (bra, ket, 0) + self.get_phh (bet, kra, 0)) / 2
                        self.set_phh (bra, ket, 0, phh)
                        self.set_phh (bet, kra, 0, phh)
                # <j|b'_q a_p|i> = <j|s-|i>
                elif np.all (hopping_index[:,bra,ket] == [-1,1]):
                    sm = (self.get_sm (bra, ket) + self.get_sm (bet, kra)) / 2
                    self.set_sm (bra, ket, sm)
                    self.set_sm (bet, kra, sm)
                # <j|b_q a_p|i>
                elif np.all (hopping_index[:,bra,ket] == [-1,-1]):
                    hh = (self.get_hh (bra, ket, 1) + self.get_hh (bet, kra, 1)) / 2
                    self.set_hh (bra, ket, 1, hh)
                    self.set_hh (bet, kra, 1, hh)
                # <j|a_q a_p|i>
                elif np.all (hopping_index[:,bra,ket] == [-2,0]):
                    hh = (self.get_hh (bra, ket, 0) + self.get_hh (bet, kra, 0)) / 2
                    self.set_hh (bra, ket, 0, hh)
                    self.set_hh (bet, kra, 0, hh)
                
        # b_p|i>
        for ket in hidx_ket_b:
            for bra in np.where ((hopping_index[1,:,ket] < 0) & idx_uniq)[0]:
                if not mask_ints[bra,ket]: continue
                bet = ptmap[bra]
                kra = ptmap[ket]
                # <j|b_p|i>
                if np.all (hopping_index[:,bra,ket] == [0,-1]):
                    h = (self.get_h (bra, ket, 1) + self.get_h (bet, kra, 1)) / 2
                    self.set_h (bra, ket, 1, h)
                    self.set_h (bet, kra, 1, h)
                    # <j|a'_q a_r b_p|i>, <j|b'_q b_r b_p|i> - how to tell if consistent sign rule?
                    if onep_index[bra,ket]:
                        phh = (self.get_phh (bra, ket, 1) + self.get_phh (bet, kra, 1)) / 2
                        self.set_phh (bra, ket, 1, phh)
                        self.set_phh (bet, kra, 1, phh)
                # <j|b_q b_p|i>
                elif np.all (hopping_index[:,bra,ket] == [0,-2]):
                    hh = (self.get_hh (bra, ket, 2) + self.get_hh (bet, kra, 2)) / 2
                    self.set_hh (bra, ket, 2, hh)
                    self.set_hh (bet, kra, 2, hh)
        
        return



    def contract_h00 (self, h_00, h_11, h_22, ket, dn=0):
        r = self.rootaddr[ket]
        n = self.fragaddr[ket]
        norb, nelec = self.norb, self.nelec_r[r]
        ci = self.ci[r][n+dn]
        h_uhf = (h_11[0] - h_11[1]) / 2
        h_uhf = [h_uhf, -h_uhf]
        h_11 = h_11.sum (0) / 2
        linkstrl = self._check_linkstrl_cache (norb, nelec[0], nelec[1])
        if h_22 is None:
            hci = h_00*ci + contract_1e (h_11, ci, norb, nelec,
                                         link_index=linkstrl)
        else:
            h2eff = absorb_h1e (h_11, h_22, norb, nelec, 0.5)
            hci = h_00*ci + contract_2e (h2eff, ci, norb, nelec,
                                         link_index=linkstrl)
        hci += contract_1e_uhf (h_uhf, ci, norb, nelec,
                                link_index=linkstrl)
        return hci

    def contract_h10 (self, spin, h_10, h_21, ket, dn=0):
        r = self.rootaddr[ket]
        n = self.fragaddr[ket]
        norb, nelec = self.norb, self.nelec_r[r]
        nelec_bra = [nelec[0], nelec[1]]
        nelec_bra[spin] += 1
        linkstrl = self._check_linkstrl_cache (norb+1, nelec_bra[0], nelec_bra[1])
        ci = self.ci[r][n+dn]
        hci = 0
        if h_21 is None or ((not isinstance (h_21, np.ndarray)) and h_21==0):
            hci = contract_1he (h_10, True, spin, ci, norb, nelec,
                                link_index=linkstrl)
        else:
            h3eff = absorb_h1he (h_10, h_21, True, spin, norb, nelec, 0.5)
            hci = contract_3he (h3eff, True, spin, ci, norb, nelec,
                                link_index=linkstrl)
        return hci

    def contract_h01 (self, spin, h_01, h_12, ket, dn=0):
        rket = self.rootaddr[ket]
        n = self.fragaddr[ket]
        norb, nelec = self.norb, self.nelec_r[rket]
        linkstrl = self._check_linkstrl_cache (norb+1, nelec[0], nelec[1])
        ci = self.ci[rket][n+dn]
        hci = 0
        if h_12 is None or ((not isinstance (h_12, np.ndarray)) and h_12==0):
            hci = contract_1he (h_01, False, spin, ci, norb, nelec,
                                link_index=linkstrl)
        else:
            h3eff = absorb_h1he (h_01, h_12, False, spin, norb, nelec, 0.5)
            hci = contract_3he (h3eff, False, spin, ci, norb, nelec,
                                link_index=linkstrl)
        return hci

    def contract_h20 (self, spin, h_20, ket, dn=0):
        r = self.rootaddr[ket]
        n = self.fragaddr[ket]
        norb, nelec = self.norb, self.nelec_r[r]
        ci = self.ci[r][n+dn]
        # 0, 1, 2 = aa, ab, bb
        s11 = int (spin>1)
        s12 = int (spin>0)
        norbd = norb + 2 - int (spin==1)
        nelecd = [n for n in nelec]
        nelecd[s11] += 1
        nelecd[s12] += 1
        linkstrl = self._check_linkstrl_cache (norbd, nelecd[0], nelecd[1])
        hci = contract_pair_op (h_20, True, spin, ci, norb, nelec, link_index=linkstrl)
        return hci

    def contract_h02 (self, spin, h_02, ket, dn=0):
        r = self.rootaddr[ket]
        n = self.fragaddr[ket]
        norb, nelec = self.norb, self.nelec_r[r]
        ci = self.ci[r][n+dn]
        # 0, 1, 2 = aa, ab, bb
        s11 = int (spin>1)
        s12 = int (spin>0)
        norbd = norb + 2 - int (spin==1)
        linkstrl = self._check_linkstrl_cache (norbd, nelec[0], nelec[1])
        hci = contract_pair_op (h_02, False, spin, ci, norb, nelec, link_index=linkstrl)
        return hci

    def contract_h11 (self, spin, h_11, ket, dn=0):
        r = self.rootaddr[ket]
        n = self.fragaddr[ket]
        norb, nelec = self.norb, self.nelec_r[r]
        ci = self.ci[r][n+dn]
        # 0, 1, 2, 3 = aa, ab, ba, bb
        s11 = spin // 2
        s12 = spin % 2
        if s11==s12:
            linkstr = self._check_linkstr_cache (norb, nelec[0], nelec[1])
            h1e = [np.zeros_like (h_11), np.zeros_like (h_11)]
            h1e[s11] = h_11
            hci = contract_1e_nosym_uhf (h1e, ci, norb, nelec, link_index=linkstr)
        else:
            linkstr = self._check_linkstr_cache (2*norb, nelec[0]+nelec[1], 0)
            spin = spin - 1
            h1e = np.zeros ((2*norb, 2*norb), dtype=h_11.dtype)
            h1e[spin*norb:(spin+1)*norb,(1-spin)*norb:(2-spin)*norb] = h_11[:,:]
            hci = contract_1e_nosym_ghf (h1e, ci, norb, nelec, link_index=linkstr)[2*spin]
        return hci

    def contract_h11_uhf (self, h_11_s, ket, dn=0):
        # when you have both spins in the operator
        r = self.rootaddr[ket]
        n = self.fragaddr[ket]
        norb, nelec = self.norb, self.nelec_r[r]
        ci = self.ci[r][n+dn]
        linkstr = self._check_linkstr_cache (norb, nelec[0], nelec[1])
        hci = contract_1e_nosym_uhf (h_11_s, ci, norb, nelec, link_index=linkstr)
        return hci

    def _init_ham_(self, nroots_si):
        self._ham = {}
        self.nroots_si = nroots_si

    def _put_ham_(self, bra, ket, h0, h1, h2, spin=None, hermi=0):
        i = self.unique_root[bra]
        j = self.unique_root[ket]
        # TODO: if you ever resurrect HamTerm.opH, you'll have to reconsider this branch
        if self.pt_order[i]>0: return # no populating the perturbative-space CI vectors
        hterm0 = self._ham.get ((i, j, hermi), 0)
        hterm1 = HamTerm (self, ket, i, j, h0, h1, h2, hermi=hermi, spin=spin)
        self._ham[(i,j,hermi)] = hterm1 + hterm0 

    def _ham_op (self, _init_only=False):
        hci_r_plab = []
        for c in self.ci:
            hci_plab = np.zeros ([self.nroots_si,] + list (c.shape),
                                 dtype=c.dtype)
            hci_r_plab.append (hci_plab)
        if not _init_only:
            for ((i, j, hermi), hterm) in self._ham.items ():
                if hterm.is_zero () or hterm.is_civec_zero (): continue
                hci_r_plab[i] += hterm.op ()
        return hci_r_plab

class HamTerm:
    def __init__(self, parent, ket, ir, jr, h0, h1, h2, hermi=0, spin=None):
        self.parent = parent
        self.ir = ir
        self.jr = jr
        self.ket = parent.rootinvaddr[jr]
        self.bra = parent.rootinvaddr[ir]
        dnelec = tuple (np.asarray (parent.nelec_r[ir]) - np.asarray (parent.nelec_r[jr]))
        self.h0 = self.h1 = self.h2 = None
        if isinstance (h1, np.ndarray):
            self.nsi, self.li, self.lj = h1.shape[:3]
        else:
            self.nsi, self.li, self.lj = h0.shape[:3]
        self.spin = spin
        self.dnelec = dnelec
        if dnelec == (0,0) and hermi==1:
            self.h0 = h0
            self.h1 = h1
            self.h2 = h2
            self._op = self._opH = parent.contract_h00
        elif dnelec == (0,0):
            spin = spin // 2
            self.h1 = np.zeros (list (h1.shape[:3]) + [2,] + list (h1.shape[3:]), dtype=h1.dtype)
            self.h1[:,:,:,spin,:,:] = h1
            self.spin = None
            self._op = self._opH = parent.contract_h11_uhf
        elif sum (dnelec) == 0:
            self.h1 = h1
            self._op = self._opH = parent.contract_h11
        else:
            dnelec = sum (dnelec)
            self.h1 = h1
            if abs (dnelec) == 1: self.h2 = h2
            idx = dnelec+2
            if idx>1: idx = idx-1
            self._op = [parent.contract_h02,
                        parent.contract_h01,
                        parent.contract_h10,
                        parent.contract_h20][idx]
            self._opH = [parent.contract_h20,
                         parent.contract_h10,
                         parent.contract_h01,
                         parent.contract_h02][idx]

    def _get_hargs (self, p, i, j):
        hargs = []
        if self.h0 is not None:
            if np.asarray (self.h0).ndim < 3:
                hargs.append (self.h0)
            else:
                hargs.append (self.h0[p,i,j])
        if self.h1 is not None:
            if np.asarray (self.h1).ndim < 3:
                hargs.append (self.h1)
            else:
                hargs.append (self.h1[p,i,j])
        if self.h2 is not None:
            if np.asarray (self.h2).ndim < 3:
                hargs.append (self.h2)
            else:
                hargs.append (self.h2[p,i,j])
        return hargs

    def op (self):
        nsi, li, lj = self.nsi, self.li, self.lj
        ndeta = self.parent.ndeta_r[self.ir]
        ndetb = self.parent.ndetb_r[self.ir]
        sargs = []
        if self.spin is not None: sargs.append (self.spin)
        hci_plab = np.zeros ((nsi,li,ndeta,ndetb), dtype=self.parent.dtype)
        if self.is_const ():
            ci = self.parent.ci[self.jr]
            if np.asarray (self.h0).ndim < 3:
                hci_plab = h0 * self.ci
            else:
                hci_plab = np.tensordot (self.h0, ci, axes=1)
        else:
            for p,i,j in product (range (nsi), range (li), range (lj)):
                if self.is_zero (idx=(p,i,j)): continue
                args = sargs + self._get_hargs (p,i,j) + [self.ket,]
                hci_plab[p,i] += self._op (*args, dn=j)
        return hci_plab

    def opH (self):
        with lib.temporary_env (self, ir=self.jr, jr=self.ir, li=self.lj, lj=self.li, ket=self.bra,
                                h0=self.get_h0H (), h1=self.get_h1H (), h2=self.get_h2H (),
                                _op=self._opH):
            return self.op ()

    def get_h0H (self):
        h0 = self.h0
        if h0 is None or np.asarray (h0).ndim < 3:
            return h0
        else:
            return h0.transpose (0,2,1).conj ()

    def get_h1H (self):
        h1 = self.h1
        if h1 is None or np.asarray (h1).ndim < 3:
            return h1
        elif self.dnelec == (0,0):
            return h1.transpose (0,2,1,3,5,4) 
        elif sum (self.dnelec)%2 == 0:
            return h1.transpose (0,2,1,4,3)
        else:
            return h1.transpose (0,2,1,3)

    def get_h2H (self):
        h2 = self.h2
        if h2 is None or np.asarray (h2).ndim < 3:
            return h2
        elif abs (sum (self.dnelec)) == 1:
            return h2.transpose (0,2,1,5,4,3)
        else:
            return h2.transpose (0,2,1,4,3,6,5)

    def __add__(self, other):
        if other==0: return self
        mysum = copy.copy (self)
        if self.h0 is not None:
            mysum.h0 = self.h0 + other.h0
        if self.h1 is not None:
            mysum.h1 = self.h1 + other.h1
        if self.h2 is not None:
            mysum.h2 = self.h2 + other.h2
        return mysum

    def is_const (self):
        h1_scalar = not isinstance (self.h1, np.ndarray)
        h2_scalar = not isinstance (self.h2, np.ndarray)
        return h1_scalar and h2_scalar and (self.h1==0) and (self.h2==0)

    def is_zero (self, idx=None):
        h0, h1, h2 = self.h0, self.h1, self.h2
        if idx is not None:
            if h0 is not None and np.asarray (h0).ndim >= 3: h0 = h0[idx]
            if h1 is not None and np.asarray (h1).ndim >= 3: h1 = h1[idx]
            if h2 is not None and np.asarray (h2).ndim >= 3: h2 = h2[idx]
        h0_zero = (h0 is None) or np.amax (np.abs (h0)) < 1e-15
        h1_zero = (h1 is None) or np.amax (np.abs (h1)) < 1e-15
        h2_zero = (h2 is None) or np.amax (np.abs (h2)) < 1e-15
        return h0_zero and h1_zero and h2_zero

    def is_civec_zero (self, conj=False):
        if conj:
            ci = self.parent.ci[self.ir]
        else:
            ci = self.parent.ci[self.jr]
        return np.amax (np.abs (ci)) < 1e-15


def make_ints (las, ci, nelec_frs, screen_linequiv=DO_SCREEN_LINEQUIV, nlas=None,
               _FragTDMInt_class=FragTDMInt, mask_ints=None, discriminator=None,
               pt_order=None, do_pt_order=None, verbose=None):
    ''' Build fragment-local intermediates (`FragTDMInt`) for LASSI o1

    Args:
        las : instance of :class:`LASCINoSymm`
        ci : list of list of ndarrays
            Contains all CI vectors
        nelec_frs : ndarray of shape (nfrags,nroots,2)
            Number of electrons of each spin in each rootspace in each
            fragment

    Kwargs:
        screen_linequiv : logical
            Whether to compress data by aggressively identifying linearly equivalent
            rootspaces and storing the relevant unitary matrices.
        mask_ints : ndarray of shape (nroots,nroots)
            Mask index down to only the included interactions
        discriminator : sequence of length (nroots)
            Additional information to descriminate between otherwise-equivalent rootspaces
        verbose : integer
            Verbosity level of intermediate logger

    Returns:
        hopping_index : ndarray of ints of shape (nfrags, 2, nroots, nroots)
            element [i,j,k,l] reports the change of number of electrons of
            spin j in fragment i between LAS rootspaces k and l
        ints : list of length nfrags of instances of :class:`FragTDMInt`
        lroots: ndarray of ints of shape (nfrags, nroots)
            Number of states within each fragment and rootspace
    '''
    nfrags, nroots = nelec_frs.shape[:2]
    log = lib.logger.new_logger (las, las.verbose)
    max_memory = getattr (las, 'max_memory', las.mol.max_memory)
    if nlas is None: nlas = las.ncas_sub
    if mask_ints is None: mask_ints = np.ones ((nroots,nroots), dtype=bool)
    lroots = get_lroots (ci)
    remaining_memory = max_memory - lib.current_memory ()[0]
    reqd_mem = lst_hopping_index_memsize (nelec_frs)
    if reqd_mem > remaining_memory:
        raise MemoryError (('lst_hopping_index requires {} MB of {} MB available ({} MB '
                            'total)').format (reqd_mem, remaining_memory, max_memory))
    hopping_index, zerop_index, onep_index = lst_hopping_index (nelec_frs)
    rootaddr, fragaddr = get_rootaddr_fragaddr (lroots)
    ints = []
    for ifrag in range (nfrags):
        m0 = lib.current_memory ()[0]
        tdmint = _FragTDMInt_class (las, ci[ifrag], hopping_index[ifrag], zerop_index, onep_index,
                                    nlas[ifrag], nroots, nelec_frs[ifrag], rootaddr,
                                    fragaddr[ifrag], ifrag, mask_ints,
                                    discriminator=discriminator,
                                    screen_linequiv=screen_linequiv,
                                    pt_order=pt_order, do_pt_order=do_pt_order,
                                    verbose=verbose)
        m1 = lib.current_memory ()[0]
        log.info ('LAS-state TDM12s fragment %d uses %f MB of %f MB total used',
                         ifrag, m1-m0, m1)
        log.timer ('LAS-state TDM12s fragment {} intermediate crunching'.format (
            ifrag), *tdmint.time_crunch)
        log.debug ('UNIQUE ROOTSPACES OF FRAG %d: %d/%d', ifrag,
                          np.count_nonzero (tdmint.root_unique), nroots)
        ints.append (tdmint)
    return hopping_index, ints, lroots


