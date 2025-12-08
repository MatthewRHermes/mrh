import numpy as np
from scipy import linalg
from pyscf import lib
from pyscf.fci.direct_spin1 import trans_rdm12s, trans_rdm1s
from pyscf.fci.direct_spin1 import contract_1e, contract_2e, absorb_h1e
from pyscf.fci.direct_uhf import contract_1e as contract_1e_uhf
from pyscf.fci.addons import cre_a, cre_b, des_a, des_b
from pyscf.fci import cistring
from itertools import product, combinations, combinations_with_replacement
from mrh.my_pyscf.lassi.citools import get_lroots, get_rootaddr_fragaddr, get_unique_roots
from mrh.my_pyscf.lassi.citools import _get_unique_roots_with_spin
from mrh.my_pyscf.lassi.op_o1.utilities import *
from mrh.my_pyscf.fci.rdm import trans_rdm1ha_des, trans_rdm1hb_des #make_rdm1_spin1
from mrh.my_pyscf.fci.rdm import trans_rdm13ha_des, trans_rdm13hb_des #is make_rdm12_spin1
from mrh.my_pyscf.fci.rdm import trans_sfddm1, trans_hhdm ##trans_sfddm1 is make_rdm12_spin1, trans_hhdm is make_rdm12_spin1
from mrh.my_pyscf.fci import rdm_smult
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
        i and j are rootspace indices

        get_h (i,j,s,**kwargs): <i|s|j>
        get_p (i,j,s,**kwargs): <i|s'|j> = conj (<j|s|i>)
        get_dm1 (i,j,**kwargs): <i|t't|j>
        get_hh (i,j,s,**kwargs): <i|s2s1|j>
        get_pp (i,j,s,**kwargs): <i|s1's2'|j> = conj (<j|s2s1|i>)
        get_sm (i,j,**kwargs): <i|b'a|j>
        get_sp (i,j,**kwargs): <i|a'b|j> = conj (<j|b'a|i>)
        get_phh (i,j,s,**kwargs): <i|t'ts|j>
        get_pph (i,j,s,**kwargs): <i|s't't|j> = conj (<j|t'ts|i>)
        get_dm2 (i,j,**kwargs): <i|t1't2't2t1|j>
        
        TODO: two-electron spin-broken components
            <i|a'b'bb|j> & h.c. & a<->b
            <i|a'a'bb|j> & a<->b
        Req'd for 2e- relativistic (i.e., spin-breaking) operators

        The optional kwargs of the get_* methods are
            highm : logical (default=False)
                If True, and if spin multiplicity information is available, returns the "high-m"
                version of the corresponding TDM with the spin vector of either the bra or the ket
                aligned with the laboratory axis
            uroot_idx : logical (default=False)
                If True, i and j are interpreted as indices into the list of *unique* rootspaces,
                not all rootspaces (see below).

        In the set_* and get_* functions, the indices i,j are rootspace indices and the major axis
        is the lroot axis. In the get_1_* functions, on the other hand, the indices i,j are single
        model state indices.

        Args:
            las : instance of :class:`LASCINoSymm` //VA: 8/18/25: seems las object is a excitationPFSCI solver?
                Only las.stdout and las.verbose (sometimes) are used to direct the logger output
            ci : list of ndarray of length nroots
                Contains CI vectors for the current fragment
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

        Some additional attributes of the class:
            nuroots : integer
                Number of unique rootspaces. If screen_linequiv==False, a rootspace is non-unique
                if it corresponds to exactly the same CI vectors as another rootspace (i.e., if the
                CI vector arrays share memory). If screen_linequiv==True, it is non-unique if its
                CI vectors span the same vector space. The discriminator can be used to distinguish
                between rootspaces that would otherwise be considered equivalent by fiat.
            uroot_idx : ndarray of length nroots
                For *all* rootspaces, indices (<nuroots) into the list of *unique* rootspaces
            uroot_addr : ndarray of length nuroots
                For *unique* rootspaces, indices (<nroots) into the list of *all* rootspaces
            nspman : integer
                Number of spin manifolds of unique rootspaces. A rootspace with good spin quantum
                number is a part of a manifold of up to smult_r[i]=2s+1 rootspaces with different
                m=(nelec_r[i][0]-nelec_r[i][1])//2, in which the CI vector are related by spin
                ladder operators.
            spman : ndarray of length nuroots
                For *unique* rootspaces, indices (<nspman) into the list of spin manifolds
            spman_inter_uroot_map : ndarray of ints of shape (nuroots, nuroots, 2)
                Map indices (i,j) to one single pair of unique rootspaces in the same spin
                manifolds and with the same mi-mj, but not necessarily the same mi and mj
                individually. Up to min (smult_r[uroot_addr[i]], smult_r[uroot_addr[j]]) pairs of
                unique rootspace might correspond to the same interaction tuple
                (spman[i],spman[j],mi-mj), but only one needs to actually be computed.
            spman_inter_uniq : ndarray of bool of shape (nuroots,nroots)
                Whether the given indices (i,j) is included among the rows of spman_inter_uroot_map
                array.
    '''

    def __init__(self, las, ci, norb, nroots, nelec_rs,
                 rootaddr, fragaddr, idx_frag, mask_ints, smult_r=None,
                 dtype=np.float64, discriminator=None,
                 pt_order=None, do_pt_order=None, screen_linequiv=DO_SCREEN_LINEQUIV,
                 verbose=None):
        # TODO: if it actually helps, cache the "linkstr" arrays
        if verbose is None: verbose = las.verbose
        if smult_r is None: smult_r = [None for n in nelec_rs]
        t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
        self.verbose = verbose
        self.log = lib.logger.new_logger (las, self.verbose)
        self.ci = ci
        self.norb = norb
        self.nroots = nroots
        self.dtype = dtype
        self.nelec_r = [tuple (n) for n in nelec_rs]
        self.spins_r = nelec_rs[:,0] - nelec_rs[:,1]
        self.smult_r = smult_r
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

    mat_keys = ('ovlp', 'h', 'hh', 'phh', 'sm', 'dm1', 'dm2')
    mdown_tdm = {'ovlp': lambda *args: args[0],
                 'h': rdm_smult.mdown_h,
                 'hh': rdm_smult.mdown_hh,
                 'phh': rdm_smult.mdown_phh,
                 'sm': rdm_smult.mdown_sm,
                 'dm1': rdm_smult.mdown_dm1,
                 'dm2': rdm_smult.mdown_dm2}
    scale_dnelec = {(-1, 0): (rdm_smult.scale_h, 0),
                    (0, -1): (rdm_smult.scale_h, 1),
                    (-2, 0): (rdm_smult.scale_hh, 0),
                    (-1, -1): (rdm_smult.scale_hh, 1),
                    (0, -2): (rdm_smult.scale_hh, 2),
                    (-1,1): (rdm_smult.scale_sm, None),
                    (0,0): (rdm_smult.scale_dm, None)}

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

    def try_get_1 (self, tag, *args, **kwargs):
        i, j = args[-2:]
        ir, jr = self.rootaddr[i], self.rootaddr[j]
        ip, jp = self.fragaddr[i], self.fragaddr[j]
        rargs = [x for x in args[:-2]] + [ir,jr]
        return self.try_get (tag, *rargs, **kwargs)[ip,jp]

    def try_get (self, tag, *args, **kwargs):
        if len (args) == 3: return self.try_get_tdm (tag, *args, **kwargs)
        elif len (args) == 2: return self.try_get_dm (tag, *args, **kwargs)
        else: raise RuntimeError (str (len (args)))

    def try_get_dm (self, tag, i, j, uroot_idx=False, highm=False):
        tab = self.mats[tag]
        mdown_fn = self.mdown_tdm[tag]
        if uroot_idx:
            ir, jr = i, j
            i, j = self.uroot_addr[i], self.uroot_addr[j]
        else:
            ir, jr = self.uroot_idx[i], self.uroot_idx[j]
        mj = self.nelec_r[j][0] - self.nelec_r[j][1]
        si, sj = self.smult_r[i], self.smult_r[j]
        try:
            ir, jr = self.spman_inter_uroot_map[ir,jr]
            assert (tab[ir][jr] is not None)
            tab = tab[ir][jr]
            if si is None: return tab
            if sj is None: return tab
            if highm: return tab
            return mdown_fn (tab, si, sj, mj)
        except Exception as e:
            errstr = 'frag {} failure to get element {},{}'.format (self.idx_frag, ir, jr)
            errstr = errstr + '\nhopping_index entry: {}'.format (self.hopping_index[:,ir,jr])
            raise RuntimeError (errstr)

    def try_get_tdm (self, tag, s, i, j, uroot_idx=False, highm=False):
        tab = self.mats[tag]
        mdown_fn = self.mdown_tdm[tag]
        if uroot_idx:
            ir, jr = i, j
            i, j = self.uroot_addr[i], self.uroot_addr[j]
        else:
            ir, jr = self.uroot_idx[i], self.uroot_idx[j]
        mj = self.nelec_r[j][0] - self.nelec_r[j][1]
        si, sj = self.smult_r[i], self.smult_r[j]
        try:
            ir, jr = self.spman_inter_uroot_map[ir,jr]
            assert (tab[s][ir][jr] is not None)
            tab = tab[s][ir][jr]
            if self.smult_r[i] is None: return tab
            if self.smult_r[j] is None: return tab
            if highm: return tab
            return mdown_fn (tab, si, s, sj, mj)
        except Exception as e:
            errstr = 'frag {} failure to get element {},{} w spin {}'.format (
                self.idx_frag, ir, jr, s)
            errstr = errstr + '\nhopping_index entry: {}'.format (self.hopping_index[:,ir,jr])
            raise RuntimeError (errstr)

    def setmanip (self, x): return np.ascontiguousarray (x)

    # 0-particle intermediate (overlap)

    @property
    def ovlp (self): return self.mats['ovlp']

    def get_ovlp (self, i, j, **kwargs):
        return self.try_get ('ovlp', i, j, **kwargs)

    def get_ovlp_inpbasis (self, i, j, **kwargs):
        ''' Apply umat if present to get the actual original-basis overlap '''
        ovlp = self.get_ovlp (i, j, **kwargs)
        if i in self.umat_root:
            ovlp = np.dot (self.umat_root[i].conj ().T, ovlp)
        if j in self.umat_root:
            ovlp = np.dot (ovlp, self.umat_root[j])
        return ovlp

    def get_1_ovlp (self, i, j, **kwargs):
        return self.try_get_1 ('ovlp', i, j, **kwargs)

    # 1-particle 1-operator intermediate

    def get_h (self, i, j, s, **kwargs):
        return self.try_get ('h', s, i, j, **kwargs)

    def set_h (self, i, j, s, x):
        i, j = self.uroot_idx[i], self.uroot_idx[j]
        x = self.setmanip (x)
        self.mats['h'][s][i][j] = x
        return x

    def get_p (self, i, j, s, **kwargs):
        return self.try_get ('h', s, j, i, **kwargs).conj ().transpose (1,0,2)

    def get_1_h (self, i, j, s, **kwargs):
        return self.try_get_1 ('h', s, i, j, **kwargs)

    def get_1_p (self, i, j, s, **kwargs):
        return self.try_get_1 ('h', s, j, i, **kwargs).conj ()

    # 2-particle intermediate

    def get_hh (self, i, j, s, **kwargs):
        return self.try_get ('hh', s, i, j, **kwargs)
        #return self.mats['hh'][s][i][j]

    def set_hh (self, i, j, s, x):
        i, j = self.uroot_idx[i], self.uroot_idx[j]
        x = self.setmanip (x)
        self.mats['hh'][s][i][j] = x
        return x

    def get_pp (self, i, j, s, **kwargs):
        return self.try_get ('hh', s, j, i, **kwargs).conj ().transpose (1,0,3,2)

    def get_1_hh (self, i, j, s, **kwargs):
        return self.try_get_1 ('hh', s, i, j, **kwargs)
        #return self.mats['hh'][s][i][j]

    def get_1_pp (self, i, j, s, **kwargs):
        return self.try_get_1 ('hh', s, j, i, **kwargs).conj ().T

    # 1-particle 3-operator intermediate
    # Note Mulliken -> Dirac transpose

    def get_phh (self, i, j, s, **kwargs):
        return self.try_get ('phh', s, i, j, **kwargs).transpose (0,1,2,4,5,3)

    def set_phh (self, i, j, s, x):
        i, j = self.uroot_idx[i], self.uroot_idx[j]
        x = self.setmanip (x)
        self.mats['phh'][s][i][j] = x
        return x

    def get_pph (self, i, j, s, **kwargs):
        return self.try_get ('phh', s, j, i, **kwargs).conj ().transpose (1,0,2,3,5,4)

    def get_1_phh (self, i, j, s, **kwargs):
        return self.try_get_1 ('phh', s, i, j, **kwargs).transpose (0,2,3,1)

    def get_1_pph (self, i, j, s, **kwargs):
        return self.try_get_1 ('phh', s, j, i, **kwargs).conj ().transpose (0,1,3,2)

    # spin-hop intermediate

    def get_sm (self, i, j, **kwargs):
        return self.try_get ('sm', i, j, **kwargs)

    def set_sm (self, i, j, x):
        i, j = self.uroot_idx[i], self.uroot_idx[j]
        x = self.setmanip (x)
        self.mats['sm'][i][j] = x
        return x

    def get_sp (self, i, j, **kwargs):
        return self.try_get ('sm', j, i, **kwargs).conj ().transpose (1,0,3,2)

    def get_1_sm (self, i, j, **kwargs):
        return self.try_get_1 ('sm', i, j, **kwargs)

    def get_1_sp (self, i, j, **kwargs):
        return self.try_get_1 ('sm', j, i, **kwargs).conj ().T

    def get_smp (self, i, j, s, **kwargs):
        if s==0: return self.get_sm (i, j, **kwargs)
        elif s==1: return self.get_sp (i, j, **kwargs)
        else: raise RuntimeError

    def get_1_smp (self, i, j, s, **kwargs):
        if s==0: return self.get_1_sm (i, j, **kwargs)
        elif s==1: return self.get_1_sp (i, j, **kwargs)
        else: raise RuntimeError

    # 1-density intermediate

    def get_dm1 (self, i, j, cs=False, **kwargs):
        k = self.uroot_idx[i]
        l = self.uroot_idx[j]
        a, b = self.spman_inter_uroot_map[k,l]
        if b > a:
            dm1 = self.try_get ('dm1', j, i, **kwargs).conj ().transpose (1,0,2,4,3)
        else:
            dm1 = self.try_get ('dm1', i, j, **kwargs)
        if cs:
            # Canonical transformation of spin d.o.f.: ab -> cs
            dm1_cs = np.zeros_like (dm1)
            dm1_cs[:,:,0] = dm1.sum (2)
            dm1_cs[:,:,1] = dm1[:,:,0] - dm1[:,:,1]
            dm1 = dm1_cs
        return dm1

    def set_dm1 (self, i, j, x):
        assert (j <= i)
        i, j = self.uroot_idx[i], self.uroot_idx[j]
        x = self.setmanip (x)
        self.mats['dm1'][i][j] = x

    def get_1_dm1 (self, i, j, **kwargs):
        k = self.uroot_idx[self.rootaddr[i]]
        l = self.uroot_idx[self.rootaddr[j]]
        a, b = self.spman_inter_uroot_map[k,l]
        if b > a:
            return self.try_get_1 ('dm1', j, i).conj ().transpose (0, 2, 1)
        return self.try_get_1 ('dm1', i, j)

    # 2-density intermediate

    def get_dm2 (self, i, j, **kwargs):
        k = self.uroot_idx[i]
        l = self.uroot_idx[j]
        a, b = self.spman_inter_uroot_map[k,l]
        if b > a:
            return self.try_get ('dm2', j, i, **kwargs).conj ().transpose (1,0,2,4,3,6,5)
        return self.try_get ('dm2', i, j, **kwargs)

    def get_1_dm2 (self, i, j, **kwargs):
        k = self.uroot_idx[self.rootaddr[i]]
        l = self.uroot_idx[self.rootaddr[j]]
        a, b = self.spman_inter_uroot_map[k,l]
        if b > a:
            return self.try_get_1 ('dm2', j, i, **kwargs).conj ().transpose (0, 2, 1, 4, 3)
        return self.try_get_1 ('dm2', i, j, **kwargs)

    def set_dm2 (self, i, j, x):
        assert (j <= i)
        i, j = self.uroot_idx[i], self.uroot_idx[j]
        x = self.setmanip (x)
        self.mats['dm2'][i][j] = x

    def get_lroots (self, i):
        return self.ci[i].shape[0]

    def get_lroots_uroot (self, i):
        return self.get_lroots (self.uroot_addr[i])

    def unmasked_int (self, i, j, screen=None):
        if self.mask_ints is not None:
            u = self.mask_ints[i,j]
        else:
            u = True
        if self.do_pt_order is not None:
            pt_order = self.pt_order[i] + self.pt_order[j]
            u = u and (pt_order in self.do_pt_order)
        if screen is not None:
            u = u and ((i in screen) or (j in screen))
        return u

    def _init_crunch_(self, screen_linequiv):
        ''' Compute the transition density matrix factors.

        Returns:
            t0 : tuple of length 2
                timestamp of entry into this function, for profiling by caller
        '''
        t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
        ci = self.ci
        ndeta, ndetb = self.ndeta_r, self.ndetb_r
        if self.mask_ints is not None:
            self.mask_ints = np.logical_or (
                self.mask_ints, self.mask_ints.T
            )

        # This is the worst-scaling (with respect to # of fragments) part of all _init_crunch_,
        # and the annoying thing is that this information was already available earlier.
        self.root_unique, self.unique_root, self.umat_root = get_unique_roots (
            ci, self.nelec_r, screen_linequiv=screen_linequiv, screen_thresh=SCREEN_THRESH,
            discriminator=self.discriminator
        )
        self.nuroots = nuroots = np.count_nonzero (self.root_unique)
        uroot_inv = -1 * np.ones (self.nroots, dtype=int)
        uroot_inv[self.root_unique] = np.arange (nuroots, dtype=int)
        self.uroot_idx = uroot_inv[self.unique_root]
        self.uroot_addr = np.where (self.root_unique)[0]
        assert (np.all (self.uroot_idx >= 0))

        self.spman = np.arange (nuroots)
        if all ([smult is not None for smult in self.smult_r]):
            ci_u = [ci[i] for i in self.uroot_addr]
            nelec_u = [self.nelec_r[i] for i in self.uroot_addr]
            smult_u = [self.smult_r[i] for i in self.uroot_addr]
            self.spman = _get_unique_roots_with_spin (ci_u, self.norb, nelec_u, smult_u)
        self.nspman = np.amax (self.spman)+1
        spman_inter = {}
        spman_inter_keys = np.empty ((nuroots,nuroots,3), dtype=int)
        for i,j in product (range (nuroots), repeat=2):
            k, l = self.uroot_addr[i], self.uroot_addr[j]
            mi = self.nelec_r[k][0] - self.nelec_r[k][1]
            mj = self.nelec_r[l][0] - self.nelec_r[l][1]
            a, b = self.spman[i], self.spman[j]
            key = (a, b, mi-mj)
            spman_inter_keys[i,j] = key
            spman_inter[key] = (i,j)
        self.spman_inter_uniq = np.zeros ((nuroots,nuroots), dtype=bool)
        self.spman_inter_uroot_map = np.empty ((nuroots,nuroots,2), dtype=int)
        for i, j in combinations_with_replacement (range (nuroots), 2):
            key = tuple (spman_inter_keys[i,j])
            p, q = spman_inter[key]
            self.spman_inter_uroot_map[i,j,:] = [p,q]
            self.spman_inter_uniq[p,q] = True
            self.spman_inter_uroot_map[j,i,:] = [q,p]
            self.spman_inter_uniq[q,p] = True

        self.mats = {}
        self.mats['ovlp'] = [[None for i in range (nuroots)] for j in range (nuroots)]
        self.mats['h'] = [[[None for i in range (nuroots)] for j in range (nuroots)] for s in (0,1)]
        self.mats['hh'] = [[[None for i in range (nuroots)] for j in range (nuroots)] for s in (-1,0,1)] 
        self.mats['phh'] = [[[None for i in range (nuroots)] for j in range (nuroots)] for s in (0,1)]
        self.mats['sm'] = [[None for i in range (nuroots)] for j in range (nuroots)]
        self.mats['dm1'] = [[None for i in range (nuroots)] for j in range (nuroots)]
        self.mats['dm2'] = [[None for i in range (nuroots)] for j in range (nuroots)]

        # Characterize the matrix elements involving these fragment states
        nelec_frs = np.asarray ([list(self.nelec_r[i]) for i in self.uroot_addr])[None,:,:]
        self.hopping_index = hopping_index = lst_hopping_index (nelec_frs)[0]
        self.hopidx_null = np.where (np.all (hopping_index==0, axis=0))
        self.hopidx_1c = [[np.where (hopping_index[s]==d) for s in (0,1)] for d in (-1,1)]
        s0 = np.array ([-1,1])[:,None,None]
        s1 = np.array ([1,-1])[:,None,None]
        self.hopidx_1s = [np.where (hopping_index==s0), np.where (hopping_index==s1)]
        s0 = np.array ([2,0])[:,None,None]
        s1 = np.array ([1,1])[:,None,None]
        s2 = np.array ([0,2])[:,None,None]
        self.hopidx_2c = [[np.where (hopping_index==c*s) for s in (s0,s1,s2)] for c in (-1,1)]

        # Update mask_ints
        if self.mask_ints is not None:
            for i in np.where (self.root_unique)[0]:
                images = np.where (self.unique_root==i)[0]
                for j in images:
                    self.mask_ints[i,:] = np.logical_or (
                        self.mask_ints[i,:], self.mask_ints[j,:]
                    )
                    self.mask_ints[:,i] = np.logical_or (
                        self.mask_ints[:,i], self.mask_ints[:,j]
                    )

        t1 = self._make_dms_()
        return t0

    def update_ci_(self, iroot, ci):
        for i, civec in zip (iroot, ci):
            assert (self.root_unique[i]), 'Cannot update non-unique CI vectors'
            self.ci[i] = civec.reshape (-1, self.ndeta_r[i], self.ndetb_r[i])
        t0 = self._make_dms_(screen=iroot)
        self.log.timer ('Update density matrices of fragment intermediate', *t0)
    

    def _trans_rdm12s_loop(self, bravecs, ketvecs, norb, nelec, linkstr):
        tdm1s = np.zeros ((bravecs.shape[0],ketvecs.shape[0],2,norb,norb), dtype=self.dtype)
        tdm2s = np.zeros ((bravecs.shape[0],ketvecs.shape[0],4,norb,norb,norb,norb),dtype=self.dtype)
        from pyscf.lib import param
        try: mgpu_fci = param.mgpu_fci
        except: mgpu_fci = False
        #try: mgpu_fci_debug = param.mgpu_fci_debug
        #except: mgpu_fci_debug = False
        #if mgpu_fci and mgpu_fci_debug:
        #  tdm1s_c = np.zeros ((bravecs.shape[0],ketvecs.shape[0],2,norb,norb), dtype=self.dtype)
        #  tdm2s_c = np.zeros ((bravecs.shape[0],ketvecs.shape[0],4,norb,norb,norb,norb),dtype=self.dtype)
        #  from gpu4mrh.fci import rdm_loops
        #  tdm1s_c, tdm2s_c = rdm_loops.trans_rdm12s(tdm1s_c, tdm2s_c, bravecs, ketvecs, norb, nelec, linkstr=linkstr)
        #  for i, j in product (range (bravecs.shape[0]), range (ketvecs.shape[0])):
        #    d1s, d2s = trans_rdm12s (bravecs[i], ketvecs[j], norb, nelec,link_index=linkstr)
        #    # Transpose based on docstring of direct_spin1.trans_rdm12s
        #    tdm1s[i,j] = np.stack (d1s, axis=0).transpose (0, 2, 1)
        #    tdm2s[i,j] = np.stack (d2s, axis=0) 
        #  tdm1s_correct = np.allclose(tdm1s, tdm1s_c)
        #  tdm2s_correct = np.allclose(tdm2s, tdm2s_c)
        #  if tdm1s_correct and tdm2s_correct:
        #    print("TDM12s correct")
        #  else: 
        #    print("TDM12s incorrect")
        #    exit()
        if mgpu_fci:
          from gpu4mrh.fci import rdm_loops
          tdm1s, tdm2s = rdm_loops.trans_rdm12s(tdm1s, tdm2s, bravecs, ketvecs, norb, nelec, linkstr=linkstr)
        else:
          for i, j in product (range (bravecs.shape[0]), range (ketvecs.shape[0])):
            d1s, d2s = trans_rdm12s (bravecs[i], ketvecs[j], norb, nelec,link_index=linkstr)
            # Transpose based on docstring of direct_spin1.trans_rdm12s
            tdm1s[i,j] = np.stack (d1s, axis=0).transpose (0, 2, 1)
            tdm2s[i,j] = np.stack (d2s, axis=0) 
        return tdm1s, tdm2s

    def _trans_rdm13h_loop(self, bravecs, ketvecs, norb, nelec_ket, spin, linkstr):
        tdm1h = np.zeros ((bravecs.shape[0],ketvecs.shape[0],norb), dtype=self.dtype)
        tdm3h = np.zeros ((bravecs.shape[0],ketvecs.shape[0],2,norb,norb,norb),dtype=self.dtype)
        from pyscf.lib import param
        try: mgpu_fci = param.mgpu_fci
        except: mgpu_fci = False
        #try: mgpu_fci_debug = param.mgpu_fci_debug
        #except: mgpu_fci_debug = False
        #if mgpu_fci and mgpu_fci_debug:
        #  tdm1h_c = np.zeros ((bravecs.shape[0],ketvecs.shape[0],norb), dtype=self.dtype)
        #  tdm3h_c = np.zeros ((bravecs.shape[0],ketvecs.shape[0],2,norb,norb,norb),dtype=self.dtype)
        #  from gpu4mrh.fci import rdm_loops
        #  tdm1h_c, tdm3h_c = rdm_loops.trans_rdm13h(tdm1h_c, tdm3h_c, bravecs, ketvecs, norb, nelec_ket, spin,linkstr)
        #  trans_rdm13h = (trans_rdm13ha_des, trans_rdm13hb_des)[spin]
        #  for i, j in product (range (bravecs.shape[0]), range (ketvecs.shape[0])):
        #    d1s, d2s = trans_rdm13h (bravecs[i], ketvecs[j], norb, nelec_ket, link_index=linkstr)
        #    tdm1h[i,j] = d1s
        #    tdm3h[i,j] = np.stack (d2s, axis=0)
        #  tdm1h_correct = np.allclose(tdm1h, tdm1h_c)
        #  tdm3h_correct = np.allclose(tdm3h, tdm3h_c)
        #  if tdm1h_correct and tdm3h_correct:
        #    print("TDM13h correct")
        #  else: 
        #    print("TDM13h incorrect")
        #    exit()
        if mgpu_fci:
          from gpu4mrh.fci import rdm_loops
          tdm1h, tdm3h = rdm_loops.trans_rdm13h(tdm1h, tdm3h, bravecs, ketvecs, norb, nelec_ket, spin,linkstr)
        else:
          trans_rdm13h = (trans_rdm13ha_des, trans_rdm13hb_des)[spin]
          for i, j in product (range (bravecs.shape[0]), range (ketvecs.shape[0])):
            d1s, d2s = trans_rdm13h (bravecs[i], ketvecs[j], norb, nelec_ket, link_index=linkstr)
            tdm1h[i,j] = d1s
            tdm3h[i,j] = np.stack (d2s, axis=0)
        return tdm1h, tdm3h

    def _trans_sfddm_loop(self, bravecs, ketvecs, norb, nelec_ket, linkstr):
        sfddm = np.zeros ((bravecs.shape[0],ketvecs.shape[0],norb,norb), dtype=self.dtype)
        from pyscf.lib import param
        try: mgpu_fci = param.mgpu_fci
        except: mgpu_fci = False
        #try: mgpu_fci_debug = param.mgpu_fci_debug
        #except: mgpu_fci_debug = False
        #if mgpu_fci and mgpu_fci_debug:
        #  from gpu4mrh.fci import rdm_loops
        #  sfddm_c = np.zeros ((bravecs.shape[0],ketvecs.shape[0],norb,norb), dtype=self.dtype)
        #  sfddm_c = rdm_loops.trans_sfddm1(sfddm_c, bravecs, ketvecs, norb, nelec_ket, linkstr=linkstr)
        #  for i, j in product (range (bravecs.shape[0]), range (ketvecs.shape[0])):
        #    d1 = trans_sfddm1 (bravecs[i], ketvecs[j], norb, nelec_ket, link_index=linkstr)
        #    sfddm[i,j] = d1
        #  sfddm_correct=np.allclose(sfddm, sfddm_c)
        #  if sfddm_correct: print("sfddm correct")
        #  else:
        #     print("sfddm incorrect")
        #     exit()
        if mgpu_fci:
          from gpu4mrh.fci import rdm_loops
          sfddm = rdm_loops.trans_sfddm1(sfddm, bravecs, ketvecs, norb, nelec_ket, linkstr=linkstr)
        else:
          for i, j in product (range (bravecs.shape[0]), range (ketvecs.shape[0])):
            d1 = trans_sfddm1 (bravecs[i], ketvecs[j], norb, nelec_ket, link_index=linkstr)
            sfddm[i,j] = d1
        
        return sfddm
    
    def _trans_hhdm_loop(self, bravecs, ketvecs, norb, nelec_ket, spin, linkstr):
        hhdm = np.zeros ((bravecs.shape[0],ketvecs.shape[0],norb,norb), dtype=self.dtype)
        from pyscf.lib import param
        try: mgpu_fci = param.mgpu_fci
        except: mgpu_fci = False
        #try: mgpu_fci_debug = param.mgpu_fci_debug
        #except: mgpu_fci_debug = False
        #if mgpu_fci and mgpu_fci_debug:
        #  from gpu4mrh.fci import rdm_loops
        #  hhdm_c = np.zeros ((bravecs.shape[0],ketvecs.shape[0],norb,norb), dtype=self.dtype)
        #  hhdm_c = rdm_loops.trans_hhdm(hhdm_c, bravecs, ketvecs, norb, nelec_ket, spin, linkstr=linkstr)
        #  for i, j in product (range (bravecs.shape[0]), range (ketvecs.shape[0])):
        #    d1 = trans_hhdm (bravecs[i], ketvecs[j], norb, nelec_ket, spin, link_index=linkstr)
        #    hhdm[i,j] = d1
        #  hhdm_correct=np.allclose(hhdm, hhdm_c)
        #  if hhdm_correct: print("hhdm correct")
        #  else:
        #     print("hhdm incorrect")
        #     exit()
        if mgpu_fci:
          from gpu4mrh.fci import rdm_loops
          hhdm = rdm_loops.trans_hhdm(hhdm, bravecs, ketvecs, norb, nelec_ket, spin, linkstr=linkstr)
        else:
          for i, j in product (range (bravecs.shape[0]), range (ketvecs.shape[0])):
            d1 = trans_hhdm (bravecs[i], ketvecs[j], norb, nelec_ket, spin, link_index=linkstr)
            hhdm[i,j] = d1
        return hhdm
    
    def _make_dms_(self, screen=None):
        t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
        t1 = (lib.logger.process_clock (), lib.logger.perf_counter ())
        ci = self.ci
        ndeta, ndetb = self.ndeta_r, self.ndetb_r
        hopping_index = self.hopping_index
        idx_uniq = self.root_unique
        spman_inter_uniq = self.spman_inter_uniq
        lroots = [c.shape[0] for c in ci]
        nroots, norb, nuroots = self.nroots, self.norb, self.nuroots
        t1 = self.log.timer_debug1 ('_make_dms_ setup', *t1)
        # Overlap matrix
        offs = np.cumsum (lroots)
        for i, j in combinations (np.where (idx_uniq)[0], 2):
            if self.nelec_r[i] != self.nelec_r[j]: continue
            if not self.unmasked_int (i,j,screen): continue
            k, l = self.uroot_idx[i], self.uroot_idx[j]
            if not (spman_inter_uniq[k,l] or spman_inter_uniq[l,k]): continue
            ci_i = ci[i].reshape (lroots[i], -1)
            ci_j = ci[j].reshape (lroots[j], -1)
            self.mats['ovlp'][k][l] = np.dot (ci_i.conj (), ci_j.T)
            self.mats['ovlp'][l][k] = self.mats['ovlp'][k][l].conj ().T
        for i in np.where (idx_uniq)[0]:
            if not self.unmasked_int (i,i,screen): continue
            ci_i = ci[i].reshape (lroots[i], -1)
            j = self.uroot_idx[i]
            self.mats['ovlp'][j][j] = np.dot (ci_i.conj (), ci_i.T)
            #errmat = self.mats['ovlp'][i][i] - np.eye (lroots[i])
            #if np.amax (np.abs (errmat)) > 1e-3:
            #    w, v = np.linalg.eigh (self.mats['ovlp'][i][i])
            #    errmsg = ('States w/in single Hilbert space must be orthonormal; '
            #              'eigvals (ovlp) = {}')
            #    raise RuntimeError (errmsg.format (w))
        t1 = self.log.timer_debug1 ('_make_dms_ overloop', *t1)


        # Loop over lroots functions
        #TODO: REFACTOR TO FARM OUT ALL TYPES OF DMS TO DIFFERENT GPUs/NODES?
        def trans_rdm12s_loop (bra_r, ket_r, do2=True):
            bravecs = ci[bra_r].reshape (-1, ndeta[bra_r], ndetb[bra_r])
            ketvecs = ci[ket_r].reshape (-1, ndeta[ket_r], ndetb[ket_r])
            nelec = self.nelec_r[ket_r]
            bravecs, ketvecs, nelec = rdm_smult.get_highm_civecs_dm (
                bravecs, ketvecs, norb, nelec, smult_bra=self.smult_r[bra_r],
                smult_ket=self.smult_r[ket_r]
            )
            linkstr = self._check_linkstr_cache (norb, nelec[0], nelec[1])
            if do2:
                tdm1s, tdm2s = self._trans_rdm12s_loop(bravecs, ketvecs, norb, nelec, linkstr)
            else:
                tdm1s = np.zeros ((bravecs.shape[0],ketvecs.shape[0],2,norb,norb), dtype=self.dtype)
                for i, j in product (range (bravecs.shape[0]), range (ketvecs.shape[0])):
                    d1s = trans_rdm1s (bravecs[i], ketvecs[j], norb, nelec,
                                       link_index=linkstr)
                    # Transpose based on docstring of direct_spin1.trans_rdm12s
                    tdm1s[i,j] = np.stack (d1s, axis=0).transpose (0, 2, 1)
            return tdm1s, tdm2s
        def trans_rdm1s_loop (bra_r, ket_r):
            return trans_rdm12s_loop (bra_r, ket_r, do2=False)[0]
        def trans_rdm13h_loop (bra_r, ket_r, spin=0, do3h=True):
            trans_rdm1h = (trans_rdm1ha_des, trans_rdm1hb_des)[spin]
            nelec_ket = self.nelec_r[ket_r]
            bravecs = ci[bra_r].reshape (-1, ndeta[bra_r], ndetb[bra_r])
            ketvecs = ci[ket_r].reshape (-1, ndeta[ket_r], ndetb[ket_r])
            bravecs, ketvecs, nelec_ket = rdm_smult.get_highm_civecs_h (
                bravecs, ketvecs, norb, nelec_ket, spin, smult_bra=self.smult_r[bra_r],
                smult_ket=self.smult_r[ket_r]
            )
            linkstr = self._check_linkstr_cache (norb+1, nelec_ket[0], nelec_ket[1])
            if do3h:
                tdm1h, tdm3h = self._trans_rdm13h_loop(bravecs, ketvecs, norb, nelec_ket, spin, linkstr)
            else:
                tdm1h = np.zeros ((bravecs.shape[0],ketvecs.shape[0],norb), dtype=self.dtype)
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
            bravecs, ketvecs, nelec_ket = rdm_smult.get_highm_civecs_sm (
                bravecs, ketvecs, norb, nelec_ket, smult_bra=self.smult_r[bra_r],
                smult_ket=self.smult_r[ket_r]
            )
            linkstr = self._check_linkstr_cache (norb+1, nelec_ket[0], nelec_ket[1]+1)
            sfddm = self._trans_sfddm_loop(bravecs, ketvecs, norb, nelec_ket, linkstr)
            return sfddm
        def trans_hhdm_loop (bra_r, ket_r, spin=0):
            bravecs = ci[bra_r].reshape (-1, ndeta[bra_r], ndetb[bra_r])
            ketvecs = ci[ket_r].reshape (-1, ndeta[ket_r], ndetb[ket_r])
            nelec_ket = self.nelec_r[ket_r]
            bravecs, ketvecs, nelec_ket = rdm_smult.get_highm_civecs_hh (
                bravecs, ketvecs, norb, nelec_ket, spin, smult_bra=self.smult_r[bra_r],
                smult_ket=self.smult_r[ket_r]
            )
            ndum = 2 - (spin%2)
            linkstr = self._check_linkstr_cache (norb+ndum, nelec_ket[0], nelec_ket[1])
            hhdm = self._trans_hhdm_loop(bravecs, ketvecs, norb, nelec_ket, spin, linkstr)
            return hhdm

        # Spectator fragment contribution
        spectator_index = np.all (hopping_index == 0, axis=0)
        spectator_index[np.triu_indices (nuroots, k=1)] = False
        spectator_index = np.stack (np.where (spectator_index), axis=1)
        for i, j in spectator_index:
            if not spman_inter_uniq[i,j]: continue
            k, l = self.uroot_addr[i], self.uroot_addr[j]
            if not self.unmasked_int (k,l,screen): continue
            #fragment is not interacting
            dm1s, dm2s = trans_rdm12s_loop (k, l, do2=True)
            self.set_dm1 (k, l, dm1s)
            self.set_dm2 (k, l, dm2s)
 
        t1 = self.log.timer_debug1 ('_make_dms_ trans_rdm12s_loop ', *t1)
        hidx_ket_a = np.where (np.any (hopping_index[0] < 0, axis=0))[0]
        hidx_ket_b = np.where (np.any (hopping_index[1] < 0, axis=0))[0]

        # a_p|i>; shape = (norb, lroots[ket], ndeta[*], ndetb[ket])
        for k in hidx_ket_a:
            for b in np.where (hopping_index[0,:,k] < 0)[0]:
                if not spman_inter_uniq[b,k]: continue
                bra, ket = self.uroot_addr[b], self.uroot_addr[k]
                if not self.unmasked_int (bra,ket,screen): continue
                # <j|a_p|i>
                if np.all (hopping_index[:,b,k] == [-1,0]):
                    h, phh = trans_rdm13h_loop (bra, ket, spin=0)
                    self.set_h (bra, ket, 0, h)
                    # <j|a'_q a_r a_p|i>, <j|b'_q b_r a_p|i> - how to tell if consistent sign rule?
                    err = np.abs (phh[:,:,0] + phh[:,:,0].transpose (0,1,4,3,2))
                    assert (np.amax (err) < 1e-8), '{}'.format (np.amax (err)) 
                    # ^ Passing this assert proves that I have the correct index
                    # and argument ordering for the call and return of trans_rdm12s
                    self.set_phh (bra, ket, 0, phh)
                    t1 = self.log.timer_debug1 ('_make_dms_ trans_rdm13h_loop ', *t1)
                # <j|b'_q a_p|i> = <j|s-|i>
                elif np.all (hopping_index[:,b,k] == [-1,1]):
                    self.set_sm (bra, ket, trans_sfddm_loop (bra, ket))
                    t1 = self.log.timer_debug1 ('_make_dms_ trans_sfddm_loop ', *t1)
                # <j|b_q a_p|i>
                elif np.all (hopping_index[:,b,k] == [-1,-1]):
                    self.set_hh (bra, ket, 1, trans_hhdm_loop (bra, ket, spin=1))
                    t1 = self.log.timer_debug1 ('_make_dms_ trans_hhdm_loop ', *t1)
                # <j|a_q a_p|i>
                elif np.all (hopping_index[:,b,k] == [-2,0]):
                    self.set_hh (bra, ket, 0, trans_hhdm_loop (bra, ket, spin=0))
                    t1 = self.log.timer_debug1 ('_make_dms_ trans_hhdm_loop ', *t1)
                
        # b_p|i>
        for k in hidx_ket_b:
            for b in np.where (hopping_index[1,:,k] < 0)[0]:
                if not spman_inter_uniq[b,k]: continue
                bra, ket = self.uroot_addr[b], self.uroot_addr[k]
                if not self.unmasked_int (bra,ket,screen): continue
                # <j|b_p|i>
                if np.all (hopping_index[:,b,k] == [0,-1]):
                    h, phh = trans_rdm13h_loop (bra, ket, spin=1)
                    self.set_h (bra, ket, 1, h)
                    # <j|a'_q a_r b_p|i>, <j|b'_q b_r b_p|i> - how to tell if consistent sign rule?
                    err = np.abs (phh[:,:,1] + phh[:,:,1].transpose (0,1,4,3,2))
                    assert (np.amax (err) < 1e-8), '{}'.format (np.amax (err))
                    # ^ Passing this assert proves that I have the correct index
                    # and argument ordering for the call and return of trans_rdm12s
                    self.set_phh (bra, ket, 1, phh)
                    t1 = self.log.timer_debug1 ('_make_dms_ trans_rdm13h_loop ', *t1)
                # <j|b_q b_p|i>
                elif np.all (hopping_index[:,b,k] == [0,-2]):
                    self.set_hh (bra, ket, 2, trans_hhdm_loop (bra, ket, spin=2))
                    t1 = self.log.timer_debug1 ('_make_dms_ trans_hhdm_loop ', *t1)
        
        return t0

    def symmetrize_pt1_(self, ptmap):
        ''' Symmetrize transition density matrices of first order in perturbation theory '''
        # TODO: memory-efficient version of this (get rid of outer product)
        mask_ints = np.add.outer (self.pt_order, self.pt_order)
        mask_ints = mask_ints==1
        mask_ints[:,self.pt_order==1] = False
        if self.mask_ints is not None:
            mask_ints = np.logical_and (
               self.mask_ints, mask_ints
            )
        ptmap = np.append (ptmap, ptmap[:,::-1], axis=0)
        ptmap = {i:j for i,j in ptmap}
        idx_uniq = self.root_unique
        for i, j in combinations (np.where (idx_uniq)[0], 2):
            if not mask_ints[i,j]: continue
            if self.nelec_r[i] != self.nelec_r[j]: continue
            k, l = ptmap[i], ptmap[j]
            o = (self.get_ovlp (i,j) + self.get_ovlp (k,l)) / 2
            i, j = self.uroot_idx[i], self.uroot_idx[j]
            k, l = self.uroot_idx[k], self.uroot_idx[l]
            self.mats['ovlp'][i][j] = self.mats['ovlp'][k][l] = o

        # Spectator fragment contribution
        hopping_index = self.hopping_index
        idx_uniq = self.root_unique
        nroots = self.nroots
        spectator_index = np.all (hopping_index == 0, axis=0)
        spectator_index = np.stack (np.where (spectator_index), axis=1)
        for i, j in spectator_index:
            i, j = self.uroot_addr[i], self.uroot_addr[j]
            if not mask_ints[i,j]: continue
            k, l = ptmap[i], ptmap[j]
            dm1s = (self.get_dm1 (i, j) + self.get_dm1 (k, l)) / 2
            self.set_dm1 (i, j, dm1s)
            if k >= l:
                self.set_dm1 (k, l, dm1s)
            else:
                self.set_dm1 (l, k, dm1s.conj ().transpose (1,0,2,4,3))
            dm2s = (self.get_dm2 (i, j) + self.get_dm2 (k, l)) / 2
            self.set_dm2 (i, j, dm2s)
            if k < l: k, l, dm2s = l, k, dm2s.conj ().transpose (1,0,2,4,3,6,5)
            self.set_dm2 (k, l, dm2s)

        hidx_ket_a = np.where (np.any (hopping_index[0] < 0, axis=0))[0]
        hidx_ket_b = np.where (np.any (hopping_index[1] < 0, axis=0))[0]

        # a_p|i>; shape = (norb, lroots[ket], ndeta[*], ndetb[ket])
        for k in hidx_ket_a:
            for b in np.where ((hopping_index[0,:,k] < 0))[0]:
                ket, bra = self.uroot_addr[k], self.uroot_addr[b]
                if not mask_ints[bra,ket]: continue
                bet = ptmap[bra]
                kra = ptmap[ket]
                # <j|a_p|i>
                if np.all (hopping_index[:,b,k] == [-1,0]):
                    h = (self.get_h (bra, ket, 0) + self.get_h (bet, kra, 0)) / 2
                    self.set_h (bra, ket, 0, h)
                    self.set_h (bet, kra, 0, h)
                    # <j|a'_q a_r a_p|i>, <j|b'_q b_r a_p|i> - how to tell if consistent sign rule?
                    phh = (self.get_phh (bra, ket, 0) + self.get_phh (bet, kra, 0)) / 2
                    # Dirac -> Mulliken transpose
                    phh = phh.transpose (0,1,2,5,3,4)
                    self.set_phh (bra, ket, 0, phh)
                    self.set_phh (bet, kra, 0, phh)
                # <j|b'_q a_p|i> = <j|s-|i>
                elif np.all (hopping_index[:,b,k] == [-1,1]):
                    sm = (self.get_sm (bra, ket) + self.get_sm (bet, kra)) / 2
                    self.set_sm (bra, ket, sm)
                    self.set_sm (bet, kra, sm)
                # <j|b_q a_p|i>
                elif np.all (hopping_index[:,b,k] == [-1,-1]):
                    hh = (self.get_hh (bra, ket, 1) + self.get_hh (bet, kra, 1)) / 2
                    self.set_hh (bra, ket, 1, hh)
                    self.set_hh (bet, kra, 1, hh)
                # <j|a_q a_p|i>
                elif np.all (hopping_index[:,b,k] == [-2,0]):
                    hh = (self.get_hh (bra, ket, 0) + self.get_hh (bet, kra, 0)) / 2
                    self.set_hh (bra, ket, 0, hh)
                    self.set_hh (bet, kra, 0, hh)
                
        # b_p|i>
        for k in hidx_ket_b:
            for b in np.where (hopping_index[1,:,k] < 0)[0]:
                ket, bra = self.uroot_addr[k], self.uroot_addr[b]
                if not mask_ints[bra,ket]: continue
                bet = ptmap[bra]
                kra = ptmap[ket]
                # <j|b_p|i>
                if np.all (hopping_index[:,b,k] == [0,-1]):
                    h = (self.get_h (bra, ket, 1) + self.get_h (bet, kra, 1)) / 2
                    self.set_h (bra, ket, 1, h)
                    self.set_h (bet, kra, 1, h)
                    # <j|a'_q a_r b_p|i>, <j|b'_q b_r b_p|i> - how to tell if consistent sign rule?
                    phh = (self.get_phh (bra, ket, 1) + self.get_phh (bet, kra, 1)) / 2
                    # Dirac -> Mulliken transpose
                    phh = phh.transpose (0,1,2,5,3,4)
                    self.set_phh (bra, ket, 1, phh)
                    self.set_phh (bet, kra, 1, phh)
                # <j|b_q b_p|i>
                elif np.all (hopping_index[:,b,k] == [0,-2]):
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

    def spin_factor_constant (self, bra, ket):
        '''Return the constant (i.e., not depending on the operator) spin factor a(m(ket))
        <bra|O(comp)|ket> = a(m(ket)) * b(m(ket),comp) * <highm(bra)|O(comp)|highm(ket)>'''
        dnelec = (self.nelec_r[bra][0] - self.nelec_r[ket][0],
                  self.nelec_r[bra][1] - self.nelec_r[ket][1])
        if (dnelec[0] > 0) or ((dnelec[0] == 0) and (dnelec[1] > 0)):
            return self.spin_factor_constant (ket, bra)
        smult_bra = self.smult_r[bra]
        smult_ket = self.smult_r[ket]
        spin_ket = self.nelec_r[ket][0] - self.nelec_r[ket][1]
        fn, spin_op = self.scale_dnelec[dnelec]
        if spin_op is None:
            return fn (smult_bra, smult_ket, spin_ket)
        else:
            return fn (smult_bra, spin_op, smult_ket, spin_ket)

    def spin_factor_component (self, bra, ket, comp):
        '''Return the operator-dependent spin factor b(m(ket),comp)
        <bra|O(comp)|ket> = a(m(ket)) * b(m(ket),comp) * <highm(bra)|O(comp)|highm(ket)>
        I can get away with this level of abstraction because for a strictly two-electron
        Hamiltonian, the only values of b turn out to be 1 or m/s. So comp==0 -> 1,
        and comp==1 -> m/s.
        '''
        if (self.smult_r[bra] != self.smult_r[ket]) or (comp == 0): return 1
        s2 = float (self.smult_r[ket] - 1) + np.finfo (float).tiny
        m2 = self.nelec_r[ket][0] - self.nelec_r[ket][1]
        return m2 / s2

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


def make_ints (las, ci, nelec_frs, smult_fr=None, screen_linequiv=DO_SCREEN_LINEQUIV, nlas=None,
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
        smult_fr : ndarray of shape (nfrags,nroots)
            Spin multiplicity of each root r in each fragment f.
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
        ints : list of length nfrags of instances of :class:`FragTDMInt`
        lroots: ndarray of ints of shape (nfrags, nroots)
            Number of states within each fragment and rootspace
    '''
    t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
    nfrags, nroots = nelec_frs.shape[:2]
    print("verbose:", verbose)
    if verbose is None: verbose = las.verbose
    log = lib.logger.new_logger (las, verbose)
    max_memory = getattr (las, 'max_memory', las.mol.max_memory)
    if nlas is None: nlas = las.ncas_sub
    if smult_fr is None: smult_fr = [None for i in range (nfrags)]
    lroots = get_lroots (ci)
    rootaddr, fragaddr = get_rootaddr_fragaddr (lroots)
    ints = []

    for ifrag in range (nfrags):
        m0 = lib.current_memory ()[0]
        tdmint = _FragTDMInt_class (las, ci[ifrag],
                                    nlas[ifrag], nroots, nelec_frs[ifrag], rootaddr,
                                    fragaddr[ifrag], ifrag, mask_ints,
                                    smult_r=smult_fr[ifrag],
                                    discriminator=discriminator,
                                    screen_linequiv=screen_linequiv,
                                    pt_order=pt_order, do_pt_order=do_pt_order,
                                    verbose=verbose)
        m1 = lib.current_memory ()[0]
        log.debug ('LAS-state TDM12s fragment %d uses %f MB of %f MB total used',
                         ifrag, m1-m0, m1)
        log.timer ('LAS-state TDM12s fragment {} intermediate crunching'.format (
            ifrag), *tdmint.time_crunch)
        log.debug ('UNIQUE ROOTSPACES OF FRAG %d: %d/%d', ifrag,
                          np.count_nonzero (tdmint.root_unique), nroots)
        ints.append (tdmint)
    return ints, lroots


