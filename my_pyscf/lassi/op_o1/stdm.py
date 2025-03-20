import numpy as np
from pyscf import lib
from pyscf.lib import logger
from itertools import product, combinations
from mrh.my_pyscf.lassi.citools import get_rootaddr_fragaddr, umat_dot_1frag_
from mrh.my_pyscf.lassi.op_o1 import frag
from mrh.my_pyscf.lassi.op_o1.utilities import *

# C interface
import ctypes
from mrh.lib.helper import load_library
liblassi = load_library ('liblassi')
def c_arr (arr): return arr.ctypes.data_as(ctypes.c_void_p)
c_int = ctypes.c_int

def mask_exc_table (exc, col=0, mask_space=None):
    if mask_space is None: return np.ones (exc.shape[0], dtype=bool)
    mask_space = np.asarray (mask_space)
    if mask_space.dtype in (bool, np.bool_):
        mask_space = np.where (mask_space)[0]
    idx = np.isin (exc[:,col], mask_space)
    return idx

class LSTDM (object):
    ''' LAS state transition density matrix intermediate 2 - whole-system DMs
        Carry out multiplications such as

            <I|sp'sq|J> = <I|sp'|J> * <I|sq|J>
            <I|s1p's2p's2p s1q|J> = <I|s1p's2p's2p|J> * <I|s1q|J>

        and so forth, where `p` and `q` are on different fragments. The parent class stores the
        entire nroots-by-nroots 1- and 2-body transition density matrix arrays (see make_stdm12s
        below), which is computed and returned by calling the `kernel` method.

        The initializer categorizes all possible interactions among a set of LAS states as
        "null" (no electrons move), "1c" (one charge unit hops; cp'cq), "1s" (one spin unit hops;
        ap'bq'bp aq), "1s1c", (a coupled spin flip of between one fragment and a charge unit which
        is hopping between two other fragments; ap'br'bq ar) "2c" (every case in which two charge
        units move among any two, three, or four fragments).

        The heart of the class is "_crunch_all_", which iterates over all listed interactions,
        builds the corresponding transition density matrices, and passes them into the "_put_D1_"
        and "_put_D2_" methods, which are overwritten in child classes to make the operator or
        reduced density matrices as appropriate.

        Subclass the __init__, __??t_D?_, __add_transpose__, and kernel methods to do various
        different things which rely on LAS-state tdm12s as intermediates without cacheing the whole
        things (i.e. operators or DMs in different basis).

        Args:
            ints : list of length nfrags of instances of :class:`FragTDMInt`
                fragment-local intermediates
            nlas : list of length nfrags of integers
                numbers of active orbitals in each fragment
            hopping_index: ndarray of ints of shape (nfrags, 2, nroots, nroots)
                element [i,j,k,l] reports the change of number of electrons of
                spin j in fragment i between LAS rootspaces k and l
            lroots: ndarray of ints of shape (nfrags, nroots)
                Number of states within each fragment and rootspace

        Kwargs:
            mask_bra_space : sequence of int or mask array of shape (nroots,)
                If included, only matrix elements involving the corresponding bra rootspaces are
                computed.
            mask_ket_space : sequence of int or mask array of shape (nroots,)
                If included, only matrix elements involving the corresponding ket rootspaces are
                computed.
            dtype : instance of np.dtype
                Currently not used; TODO: generalize to ms-broken fragment-local states?
        '''
    # TODO: SO-LASSI o1 implementation: a SOMF implementation using spin-pure LAS product states
    # states as a basis requires the sz-breaking sector of the 1-body stdm1 to be added here. I.E.,
    # in addition to the interactions listed above, we also need "sm" (total spin lowering; ap'bq)
    # (N.B.: "sp" is just the adjoint of "sm"). 
    # TODO: at some point, if it ever becomes rate-limiting, make this multithread better

    def __init__(self, ints, nlas, hopping_index, lroots, mask_bra_space=None, mask_ket_space=None,
                 log=None, max_memory=2000, dtype=np.float64):
        self.ints = ints
        self.log = log
        self.max_memory = max_memory
        self.nlas = nlas
        self.norb = sum (nlas)
        self.lroots = lroots
        self.rootaddr, self.envaddr = get_rootaddr_fragaddr (lroots)
        self.envaddr = np.ascontiguousarray (self.envaddr.T)
        nprods = np.prod (lroots, axis=0)
        offs1 = np.cumsum (nprods)
        offs0 = offs1 - nprods
        self.offs_lroots = np.stack ([offs0, offs1], axis=1)
        self.nfrags, _, self.nroots, _ = hopping_index.shape
        self.strides = np.append (np.ones (self.nroots, dtype=int)[None,:],
                                  np.cumprod (lroots[:-1,:], axis=0),
                                  axis=0).T
        self.strides = np.ascontiguousarray (self.strides)
        self.nstates = offs1[-1]
        self.dtype = dtype
        self.tdm1s = self.tdm2s = None

        # overlap tensor
        self.ovlp = [i.ovlp for i in ints]

        # spin-shuffle sign vector
        self.nelec_rf = np.asarray ([[list (i.nelec_r[ket]) for i in ints]
                                     for ket in range (self.nroots)]).transpose (0,2,1)
        self.spin_shuffle = [fermion_spin_shuffle (nelec_sf[0], nelec_sf[1])
                             for nelec_sf in self.nelec_rf]
        self.nelec_rf = self.nelec_rf.sum (1)

        self.urootstr = np.asarray ([[i.unique_root[r] for i in self.ints]
                                     for r in range (self.nroots)]).T

        exc = self.make_exc_tables (hopping_index)
        self.nonuniq_exc = {}
        self.exc_null = self.mask_exc_table_(exc['null'], 'null', mask_bra_space, mask_ket_space)
        self.exc_1d = self.mask_exc_table_(exc['1d'], '1d', mask_bra_space, mask_ket_space)
        self.exc_2d = self.mask_exc_table_(exc['2d'], '2d', mask_bra_space, mask_ket_space)
        self.exc_1c = self.mask_exc_table_(exc['1c'], '1c', mask_bra_space, mask_ket_space)
        self.exc_1c1d = self.mask_exc_table_(exc['1c1d'], '1c1d', mask_bra_space, mask_ket_space)
        self.exc_1s = self.mask_exc_table_(exc['1s'], '1s', mask_bra_space, mask_ket_space)
        self.exc_1s1c = self.mask_exc_table_(exc['1s1c'], '1s1c', mask_bra_space, mask_ket_space)
        self.exc_2c = self.mask_exc_table_(exc['2c'], '2c', mask_bra_space, mask_ket_space)
        self.init_profiling ()

        # buffer
        bigorb = np.sort (nlas)[::-1]
        self.d1 = np.zeros (2*(sum(bigorb[:2])**2), dtype=self.dtype)
        self.d2 = np.zeros (4*(sum(bigorb[:4])**4), dtype=self.dtype)
        self._norb_c = c_int (self.norb)
        self._orbidx = np.ones (self.norb, dtype=bool)

        # C fns
        if self.dtype==np.float64:
            self._put_SD1_c_fn = liblassi.LASSIRDMdputSD1
            self._put_SD2_c_fn = liblassi.LASSIRDMdputSD2
        elif self.dtype==np.complex128:
            self._put_SD1_c_fn = liblassi.LASSIRDMzputSD1
            self._put_SD2_c_fn = liblassi.LASSIRDMzputSD2
        else:
            raise NotImplementedError (self.dtype)

    def interaction_fprint (self, bra, ket, frags, ltri=False):
        frags = np.sort (frags)
        brastr = self.urootstr[frags,bra]
        ketstr = self.urootstr[frags,ket]
        if ltri: brastr, ketstr = sorted ([list(brastr),list(ketstr)])
        fprint = np.stack ([frags, brastr, ketstr], axis=0)
        return fprint

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
        self.dt_g, self.dw_g = 0.0, 0.0
        self.dt_s, self.dw_s = 0.0, 0.0

    # This needs to be changed in hci constant-part subclass
    def fermion_frag_shuffle (self, iroot, frags):
        return fermion_frag_shuffle (self.nelec_rf[iroot], frags)

    def make_exc_tables (self, hopping_index):
        ''' Generate excitation tables. The nth column of each array is the (n+1)th argument of the
        corresponding _crunch_*_ member function below. The first two columns are always the bra
        rootspace index and the ket rootspace index, respectively. Further columns identify
        fragments whose quantum numbers are changed by the interaction. If necessary (i.e., for 1c
        and 2c), the last column identifies spin case.

        Args:
            hopping_index: ndarray of ints of shape (nfrags, 2, nroots, nroots)
                element [i,j,k,l] reports the change of number of electrons of
                spin j in fragment i between LAS rootspaces k and l

        Returns:
            exc: dict with str keys and ndarray-of-int values. Each row of each ndarray is the
                argument list for 1 call to the LSTDM._crunch_*_ function with the name that
                corresponds to the key str (_crunch_1d_, _crunch_1s_, etc.).
        '''
        exc = {}
        exc['null'] = np.empty ((0,2), dtype=int)
        exc['1d'] = np.empty ((0,3), dtype=int)
        exc['2d'] = np.empty ((0,4), dtype=int)
        exc['1c'] = np.empty ((0,5), dtype=int)
        exc['1c1d'] = np.empty ((0,6), dtype=int)
        exc['1s'] = np.empty ((0,4), dtype=int)
        exc['1s1c'] = np.empty ((0,5), dtype=int)
        exc['1s1c_T'] = np.empty ((0,5), dtype=int)
        exc['2c'] = np.empty ((0,7), dtype=int)
        nfrags = hopping_index.shape[0]

        # Process connectivity data to quickly distinguish interactions

        # Should probably be all == true anyway if I call this by symmetry blocks
        conserv_index = np.all (hopping_index.sum (0) == 0, axis=0)

        # Number of field operators involved in a given interaction
        nsop = np.abs (hopping_index).sum (0) # 0,0 , 2,0 , 0,2 , 2,2 , 4,0 , 0,4
        nop = nsop.sum (0) # 0, 2, 4
        ispin = nsop[1,:,:] // 2
        # This last ^ is somewhat magical, but notice that it corresponds to the mapping
        #   2,0 ; 4,0 -> 0 -> a or aa
        #   0,2 ; 2,2 -> 1 -> b or ab
        #   0,4       -> 2 -> bb

        # For each interaction, the change to each fragment of
        charge_index = hopping_index.sum (1) # charge
        spin_index = hopping_index[:,0] - hopping_index[:,1] # spin (*2)

        # Upon a given interaction, count the number of fragments which:
        ncharge_index = np.count_nonzero (charge_index, axis=0) # change in charge
        nspin_index = np.count_nonzero (spin_index, axis=0) # change in spin

        findf = np.argsort ((3*hopping_index[:,0]) + hopping_index[:,1], axis=0, kind='stable')
        # This is an array of shape (nfrags, nroots, nroots) such that findf[:,i,j]
        # is list of fragment indices sorted first by the number of spin-up electrons
        # gained (>0) or lost (<0), and then by the number of spin-down electrons gained
        # or lost in the interaction between states "i" and "j". Because at most 2
        # des/creation ops are involved, the factor of 3 sets up the order a'b'ba without
        # creating confusion between spin and charge of freedom. The 'stable' sort keeps
        # relative order -> sign convention!
        #
        # Throughout the below, we use element-wise logical operators to generate mask
        # index arrays addressing elements of the last two dimensions of "findf" that
        # are consistent with a state interaction of a specific type. We then use the
        # fragment index lists thus specified to identify the source and destination
        # fragments of the charge or spin units that are transferred in that interaction,
        # and store those fragment indices along with the state indices.

        # Zero-electron interactions
        tril_index = np.zeros_like (conserv_index)
        tril_index[np.tril_indices (self.nroots)] = True
        idx = conserv_index & tril_index & (nop == 0)
        exc['null'] = np.vstack (list (np.where (idx))).T
 
        # One-density interactions
        fragrng = np.arange (nfrags, dtype=int)
        exc['1d'] = np.append (np.repeat (exc['null'], nfrags, axis=0),
                               np.tile (fragrng, len (exc['null']))[:,None],
                               axis=1)

        # Two-density interactions
        if nfrags > 1:
            fragrng = np.stack (np.tril_indices (nfrags, k=-1), axis=1)
            exc['2d'] = np.append (np.repeat (exc['null'], len (fragrng), axis=0),
                                   np.tile (fragrng, (len (exc['null']), 1)),
                                   axis=1)

        # One-electron interactions
        idx = conserv_index & (nop == 2) & tril_index
        if nfrags > 1: exc['1c'] = np.vstack (
            list (np.where (idx)) + [findf[-1][idx], findf[0][idx], ispin[idx]]
        ).T

        # One-electron, one-density interactions
        if nfrags > 2:
            fragrng = np.arange (nfrags, dtype=int)
            exc['1c1d'] = np.append (np.repeat (exc['1c'], nfrags, axis=0),
                                     np.tile (fragrng, len (exc['1c']))[:,None],
                                     axis=1)
            invalid = ((exc['1c1d'][:,2] == exc['1c1d'][:,5])
                       | (exc['1c1d'][:,3] == exc['1c1d'][:,5]))
            exc['1c1d'] = exc['1c1d'][~invalid,:][:,[0,1,2,3,5,4]]

        # Unsymmetric two-electron interactions: full square
        idx_2e = conserv_index & (nop == 4)

        # Two-electron interaction: ii -> jk ("split").
        idx = idx_2e & (ncharge_index == 3) & (np.amin (charge_index, axis=0) == -2)
        if nfrags > 2: exc_split = np.vstack (
            list (np.where (idx)) + [findf[-1][idx], findf[0][idx], findf[-2][idx], findf[0][idx],
            ispin[idx]]
        ).T

        # Two-electron interaction: ij -> kk ("coalesce")
        idx = idx_2e & (ncharge_index == 3) & (np.amax (charge_index, axis=0) == 2)
        if nfrags > 2: exc_coalesce = np.vstack (
            list (np.where (idx)) + [findf[-1][idx], findf[0][idx], findf[-1][idx], findf[1][idx],
            ispin[idx]]
        ).T

        # Two-electron interaction: k(a)j(b) -> i(a)k(b) ("1s1c")
        idx = idx_2e & (nspin_index==3) & (ncharge_index==2) & (np.amin(spin_index,axis=0)==-2)
        if nfrags > 2: exc['1s1c'] = np.vstack (
            list (np.where (idx)) + [findf[-1][idx], findf[1][idx], findf[0][idx]]
        ).T

        # Two-electron interaction: k(b)j(a) -> i(b)k(a) ("1s1c_T")
        # This will only be used when we are unable to restrict ourselves to the lower triangle
        idx = idx_2e & (nspin_index==3) & (ncharge_index==2) & (np.amax(spin_index,axis=0)==2)
        if nfrags > 2: exc_1s1cT = np.vstack (
            list (np.where (idx)) + [findf[-2][idx], findf[0][idx], findf[-1][idx]]
        ).T

        # Symmetric two-electron interactions: lower triangle only
        idx_2e = idx_2e & tril_index

        # Two-electron interaction: i(a)j(b) -> j(a)i(b) ("1s") 
        idx = idx_2e & (ncharge_index == 0) & (nspin_index == 2)
        if nfrags > 1: exc['1s'] = np.vstack (
            list (np.where (idx)) + [findf[-1][idx], findf[0][idx]]
        ).T

        # Two-electron interaction: ii -> jj ("pair") 
        idx = idx_2e & (ncharge_index == 2) & (nspin_index < 3)
        if nfrags > 1: exc_pair = np.vstack (
            list (np.where (idx)) + [findf[-1][idx], findf[0][idx], findf[-1][idx], findf[0][idx],
            ispin[idx]]
        ).T

        # Two-electron interaction: ij -> kl ("scatter")
        idx = idx_2e & (ncharge_index == 4)
        if nfrags > 3: exc_scatter = np.vstack (
            list (np.where (idx)) + [findf[-1][idx], findf[0][idx], findf[-2][idx], findf[1][idx], 
            ispin[idx]]
        ).T

        if self.all_interactions_full_square and nfrags > 2:
            exc['1s1c'] = np.append (
                np.pad (exc['1s1c'], ((0,0),(0,1)), constant_values=0),
                np.pad (exc_1s1cT,   ((0,0),(0,1)), constant_values=1),
                axis=0)
            exc_split = np.append (exc_split, exc_coalesce, axis=0)

        # Combine "split", "pair", and "scatter" into "2c"
        if nfrags > 1: exc['2c'] = exc_pair
        if nfrags > 2: exc['2c'] = np.vstack ((exc['2c'], exc_split))
        if nfrags > 3: exc['2c'] = np.vstack ((exc['2c'], exc_scatter))

        return exc

    all_interactions_full_square = False
    interaction_has_spin = ('_1c_', '_1c1d_', '_2c_')
    ltri_ambiguous = True

    def mask_exc_table_(self, exc, lbl, mask_bra_space=None, mask_ket_space=None):
        # Part 1: restrict to the caller-specified rectangle
        idx  = mask_exc_table (exc, col=0, mask_space=mask_bra_space)
        idx &= mask_exc_table (exc, col=1, mask_space=mask_ket_space)
        exc = exc[idx]
        # Part 2: identify interactions which are equivalent except for the overlap
        # factor of spectator fragments. Reduce the exc table only to the unique
        # interactions and populate self.nonuniq_exc with the corresponding
        # nonunique images.
        if lbl=='null': return exc
        ulblu = '_' + lbl + '_'
        excp = exc[:,:-1] if ulblu in self.interaction_has_spin else exc
        fprintLT = []
        fprint = []
        for row in excp:
            bra, ket = row[:2]
            frags = row[2:]
            fpLT = self.interaction_fprint (bra, ket, frags, ltri=self.ltri_ambiguous)
            fprintLT.append (fpLT.ravel ())
            fp = self.interaction_fprint (bra, ket, frags, ltri=False)
            fprint.append (fp.ravel ())
        fprintLT = np.asarray (fprintLT)
        fprint = np.asarray (fprint)
        nexc = len (exc)
        fprintLT, idx, inv = np.unique (fprintLT, axis=0, return_index=True, return_inverse=True)
        # for some reason this squeeze is necessary for some versions of numpy; however...
        eqmap = np.squeeze (idx[inv])
        for fpLT, uniq_idx in zip (fprintLT, idx):
            row_uniq = excp[uniq_idx]
            # ...numpy.where (0==0) triggers a DeprecationWarning, so I have to atleast_1d it
            uniq_idxs = np.where (np.atleast_1d (eqmap==uniq_idx))[0]
            braket_images = exc[np.ix_(uniq_idxs,[0,1])]
            iT = np.any (fprint[uniq_idx][None,:]!=fprint[uniq_idxs], axis=1)
            braket_images[iT,:] = braket_images[iT,::-1]
            self.nonuniq_exc[tuple(row_uniq)] = braket_images
        exc = exc[idx]
        nuniq = len (exc)
        self.log.debug ('%d/%d unique interactions of %s type',
                        nuniq, nexc, lbl)
        return exc

    def get_range (self, i):
        '''Get the orbital range for a fragment.

        Args:
            i: integer
                index of a fragment

        Returns:
            p: integer
                beginning of ith fragment orbital range
            q: integer
                end of ith fragment orbital range
        '''
        p = sum (self.nlas[:i])
        q = p + self.nlas[i]
        return p, q

    def get_ovlp_fac (self, bra, ket, *inv):
        '''Compute the overlap * permutation factor between two model states for a given list of
        non-spectator fragments.

        Args:
            bra: integer
                Index of a model state
            ket: integer
                Index of a model state
            *inv: integers
                Indices of nonspectator fragments

        Returns:
            wgt: float
                The product of the overlap matrix elements between bra and ket for all fragments
                not included in *inv, multiplied by the fermion permutation factor required to
                bring the field operators of those in *inv adjacent to each other in normal
                order.
        '''
        idx = np.ones (self.nfrags, dtype=np.bool_)
        idx[list (inv)] = False
        wgt = np.prod ([i.get_1_ovlp (bra, ket) for i, ix in zip (self.ints, idx) if ix])
        uniq_frags = list (set (inv))
        bra, ket = self.rootaddr[bra], self.rootaddr[ket]
        wgt *= self.spin_shuffle[bra] * self.spin_shuffle[ket]
        wgt *= self.fermion_frag_shuffle (bra, uniq_frags)
        wgt *= self.fermion_frag_shuffle (ket, uniq_frags)
        return wgt

    def crunch_ovlp (self, bra, ket):
        i = self.ints[-1]
        b, k = i.unique_root[bra], i.unique_root[ket]
        o = i.ovlp[b][k] / (1 + int (bra==ket))
        for i in self.ints[-2::-1]:
            o = np.multiply.outer (o, i.get_ovlp (bra, ket)).transpose (0,2,1,3)
            o = o.reshape (o.shape[0]*o.shape[1], o.shape[2]*o.shape[3])
        o *= self.spin_shuffle[bra]
        o *= self.spin_shuffle[ket]
        return o

    def get_ci_dtype (self):
        for inti in self.ints:
            if inti.dtype == np.complex128: return np.complex128
        return np.float64

    def get_ovlp (self, rootidx=None):
        lroots = self.lroots.copy ()
        exc_null = self.exc_null
        offs_lroots = self.offs_lroots.copy ()
        nstates = self.nstates
        if rootidx is not None:
            rootidx = np.atleast_1d (rootidx)
            bra_null = np.isin (self.exc_null[:,0], rootidx)
            ket_null = np.isin (self.exc_null[:,1], rootidx)
            exc_null = exc_null[bra_null&ket_null,:]
            lroots = lroots[:,rootidx]
            nprods = np.prod (lroots, axis=0)
            offs1 = np.cumsum (nprods)
            offs0 = offs1 - nprods
            for i, iroot in enumerate (rootidx):
                offs_lroots[iroot,:] = [offs0[i], offs1[i]]
            nstates = offs1[-1]
        ovlp = np.zeros ([nstates,]*2, dtype=self.get_ci_dtype ())
        for bra, ket in exc_null:
            i0, i1 = offs_lroots[bra]
            j0, j1 = offs_lroots[ket]
            ovlp[i0:i1,j0:j1] = self.crunch_ovlp (bra, ket)
        ovlp += ovlp.T
        for ifrag, inti in enumerate (self.ints):
            iroot_ids = iroot_poss = inti.umat_root.keys ()
            if rootidx is not None:
                iroot_ids = np.asarray (list (iroot_ids))
                idx = np.isin (rootidx, iroot_ids)
                iroot_ids = rootidx[idx]
                iroot_poss = np.where (idx)[0]
            for iroot_id, iroot_pos in zip (iroot_ids, iroot_poss):
                umat = inti.umat_root[iroot_id]
                ovlp = umat_dot_1frag_(ovlp, umat, lroots, ifrag, iroot_pos, axis=0)
                ovlp = umat_dot_1frag_(ovlp, umat, lroots, ifrag, iroot_pos, axis=1)
        return ovlp 

    def _get_addr_range (self, raddr, *inv, _profile=True):
        '''Get the integer offsets for successive ENVs in a particular rootspace in which some
        fragments are frozen in the zero state.

        Args:
            raddr: integer
                Index of a rootspace
            *inv: integers
                Indicies of fragments to be included in the iteration. All other fragments are
                frozen in the zero state.

        Returns
            addrs: ndarray of integers
                Indices of states with different excitation numbers in the fragments in *inv, with
                all other fragments frozen in the zero state.
        '''
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        addr0, addr1 = self.offs_lroots[raddr]
        inv = list (set (inv))
        lroots = self.lroots[:,raddr:raddr+1]
        envaddr_inv = get_rootaddr_fragaddr (lroots[inv])[1]
        strides_inv = self.strides[raddr][inv]
        addrs = addr0 + np.dot (strides_inv, envaddr_inv)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        if _profile: self.dt_i, self.dw_i = self.dt_i + dt, self.dw_i + dw
        return addrs

    def _prepare_spec_addr_ovlp_(self, rbra, rket, *inv):
        '''Prepare the cache for _get_spec_addr_ovlp.

        Args:
            rbra: integer
                Index of bra rootspace for which to prepare the current cache.
            rket: integer
                Index of ket rootspace for which to prepare the current cache.
            *inv: integers
                Indices of nonspectator fragments
        '''
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        key = tuple ((rbra,rket)) + inv
        braket_table = self.nonuniq_exc[key]
        self._spec_addr_ovlp_cache = []
        for rbra1, rket1 in braket_table:
            b, k, o = self._get_spec_addr_ovlp_1space (rbra1, rket1, *inv)
            self._spec_addr_ovlp_cache.append ((rbra1, rket1, b, k, o))
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_o, self.dw_o = self.dt_o + dt, self.dw_o + dw
        return

    def _get_spec_addr_ovlp (self, bra, ket, *inv):
        '''Obtain the integer indices and overlap*permutation factors for all pairs of model states
        for which a specified list of nonspectator fragments are in same state that they are in a
        provided input pair bra, ket. Uses a cache that must be prepared beforehand by the function
        _prepare_spec_addr_ovlp_(rbra, rket, *inv), where rbra and rket must be the rootspace
        indices corresponding to this function's bra, ket arguments.

        Args:
            bra: integer
                Index of a model state
            ket: integer
                Index of a model state
            *inv: integers
                Indices of nonspectator fragments.

        Returns:
            bra_rng: ndarray of integers
                Indices of model states in which fragments *inv have the same state as bra
            ket_rng: ndarray of integers
                Indices of model states in which fragments *inv have the same state as ket
            facs: ndarray of floats
                Overlap * permutation factors (cf. get_ovlp_fac) corresponding to the interactions
                bra_rng, ket_rng.
        '''
        # NOTE: from tests on triene 3frag LASSI[3,3], this function is 1/4 to 1/6 of the "put"
        # runtime, and apparently it can sometimes multithread somehow???
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        rbra, rket = self.rootaddr[bra], self.rootaddr[ket]
        braenv = self.envaddr[bra]
        ketenv = self.envaddr[ket]
        bra_rng = []
        ket_rng = []
        facs = []
        for (rbra1, rket1, b, k, o) in self._spec_addr_ovlp_cache:
            dbra = np.dot (braenv, self.strides[rbra1])
            dket = np.dot (ketenv, self.strides[rket1])
            bra_rng.append (b+dbra)
            ket_rng.append (k+dket)
            facs.append (o)
        bra_rng = np.concatenate (bra_rng)
        ket_rng = np.concatenate (ket_rng)
        facs = np.concatenate (facs)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_g, self.dw_g = self.dt_g + dt, self.dw_g + dw
        return bra_rng, ket_rng, facs

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
                Indices of model states in the rootspace of bra in which fragments *inv are in the
                zero state
            ket_rng: ndarray of integers
                Indices of model states in the rootspace of ket in which fragments *inv are in the
                zero_state
            o: ndarray of floats
                Overlap * permutation factors (cf. get_ovlp_fac) corresponding to the interactions
                bra_rng, ket_rng.
        '''
        inv = list (set (inv))
        fac = self.spin_shuffle[rbra] * self.spin_shuffle[rket]
        fac *= self.fermion_frag_shuffle (rbra, inv)
        fac *= self.fermion_frag_shuffle (rket, inv)
        spec = np.ones (self.nfrags, dtype=bool)
        for i in inv: spec[i] = False
        spec = np.where (spec)[0]
        bra_rng = self._get_addr_range (rbra, *spec, _profile=False)
        ket_rng = self._get_addr_range (rket, *spec, _profile=False)
        specints = [self.ints[i] for i in spec]
        o = fac * np.ones ((1,1), dtype=self.get_ci_dtype ())
        for i in specints:
            o = np.multiply.outer (i.get_ovlp (rbra, rket), o).transpose (0,2,1,3)
            o = o.reshape (o.shape[0]*o.shape[1], o.shape[2]*o.shape[3])
        idx = np.abs(o) > 1e-8
        if (rbra==rket): # not bra==ket because _loop_lroots_ doesn't restrict to tril
            o[np.diag_indices_from (o)] *= 0.5
            idx[np.triu_indices_from (idx, k=1)] = False
        o = o[idx]
        idx, idy = np.where (idx)
        bra_rng, ket_rng = bra_rng[idx], ket_rng[idy]
        return bra_rng, ket_rng, o

    def _get_D1_(self, bra, ket):
        self.d1[:] = 0.0
        return self.d1

    def _get_D2_(self, bra, ket):
        self.d2[:] = 0.0
        return self.d2

    def _put_D1_(self, bra, ket, D1, *inv):
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        bra1, ket1, wgt = self._get_spec_addr_ovlp (bra, ket, *inv)
        self._put_SD1_(bra1, ket1, D1, wgt)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_p, self.dw_p = self.dt_p + dt, self.dw_p + dw

    def _put_SD1_(self, bra, ket, D1, wgt):
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        #idx = self._orbidx
        #idx = np.ix_([True,]*2,idx,idx)
        #for b, k, w in zip (bra, ket, wgt):
        fn = self._put_SD1_c_fn
        c_one = c_int (1)
        for b, k, w in zip (bra, ket, wgt):
            D1w = D1 * w
            fn (c_arr (self.tdm1s[b,k]), c_arr (D1w),
                c_one, self._norb_c, self._nsrc_c,
                self._dblk_idx, self._sblk_idx, self._lblk, self._nblk)
        #    self.tdm1s[b,k][idx] += np.multiply.outer (w, D1)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_s, self.dw_s = self.dt_s + dt, self.dw_s + dw

    def _put_D2_(self, bra, ket, D2, *inv):
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        bra1, ket1, wgt = self._get_spec_addr_ovlp (bra, ket, *inv)
        self._put_SD2_(bra1, ket1, D2, wgt)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_p, self.dw_p = self.dt_p + dt, self.dw_p + dw

    def _put_SD2_(self, bra, ket, D2, wgt):
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        #idx = self._orbidx
        #idx = np.ix_([True,]*4,idx,idx,idx,idx)
        #for b, k, w in zip (bra, ket, wgt):
        #    self.tdm2s[b,k][idx] += np.multiply.outer (w, D2)
        fn = self._put_SD2_c_fn
        c_one = c_int (1)
        for b, k, w in zip (bra, ket, wgt):
            D2w = D2 * w
            fn (c_arr (self.tdm2s[b,k]), c_arr (D2w),
                c_one, self._norb_c, self._nsrc_c, self._pdest,
                self._dblk_idx, self._sblk_idx, self._lblk, self._nblk)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_s, self.dw_s = self.dt_s + dt, self.dw_s + dw

    # Cruncher functions
    def _crunch_1d_(self, bra, ket, i):
        '''Compute a single-fragment density fluctuation, for both the 1- and 2-RDMs.'''
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        d1 = self._get_D1_(bra, ket)
        d2 = self._get_D2_(bra, ket)
        p, q = self.get_range (i)
        inti = self.ints[i]
        d1_s_ii = inti.get_1_dm1 (bra, ket)
        d1[:,p:q,p:q] = np.asarray (d1_s_ii)
        d2[:,p:q,p:q,p:q,p:q] = np.asarray (inti.get_1_dm2 (bra, ket))
        self._put_D1_(bra, ket, d1, i)
        self._put_D2_(bra, ket, d2, i)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_1d, self.dw_1d = self.dt_1d + dt, self.dw_1d + dw

    def _crunch_2d_(self, bra, ket, i, j):
        '''Compute a two-fragment density fluctuation.'''
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        d2 = self._get_D2_(bra, ket)
        inti, intj = self.ints[i], self.ints[j]
        p, q = self.get_range (i)
        r, s = self.get_range (j)
        d1_s_ii = inti.get_1_dm1 (bra, ket)
        d1_s_jj = intj.get_1_dm1 (bra, ket)
        d2_s_iijj = np.multiply.outer (d1_s_ii, d1_s_jj).transpose (0,3,1,2,4,5)
        d2_s_iijj = d2_s_iijj.reshape (4, q-p, q-p, s-r, s-r)
        d2[:,p:q,p:q,r:s,r:s] = d2_s_iijj
        d2[(0,3),r:s,r:s,p:q,p:q] = d2_s_iijj[(0,3),...].transpose (0,3,4,1,2)
        d2[(1,2),r:s,r:s,p:q,p:q] = d2_s_iijj[(2,1),...].transpose (0,3,4,1,2)
        d2[(0,3),p:q,r:s,r:s,p:q] = -d2_s_iijj[(0,3),...].transpose (0,1,4,3,2)
        d2[(0,3),r:s,p:q,p:q,r:s] = -d2_s_iijj[(0,3),...].transpose (0,3,2,1,4)
        self._put_D2_(bra, ket, d2, i, j)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_2d, self.dw_2d = self.dt_2d + dt, self.dw_2d + dw

    def _crunch_1c_(self, bra, ket, i, j, s1):
        '''Compute the reduced density matrix elements of a single electron hop; i.e.,

        <bra|j'(s1)i(s1)|ket>

        i.e.,

        j ---s1---> i

        and conjugate transpose
        '''
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        d1 = self._get_D1_(bra, ket)
        d2 = self._get_D2_(bra, ket)
        inti, intj = self.ints[i], self.ints[j]
        p, q = self.get_range (i)
        r, s = self.get_range (j)
        fac = 1
        nelec_f_bra = self.nelec_rf[self.rootaddr[bra]]
        nelec_f_ket = self.nelec_rf[self.rootaddr[ket]]
        fac *= fermion_des_shuffle (nelec_f_bra, (i, j), i)
        fac *= fermion_des_shuffle (nelec_f_ket, (i, j), j)
        d1_ij = np.multiply.outer (self.ints[i].get_1_p (bra, ket, s1),
                                   self.ints[j].get_1_h (bra, ket, s1))
        d1[s1,p:q,r:s] = fac * d1_ij
        s12l = s1 * 2   # aa: 0 OR ba: 2
        s12h = s12l + 1 # ab: 1 OR bb: 3 
        s21l = s1       # aa: 0 OR ab: 1
        s21h = s21l + 2 # ba: 2 OR bb: 3
        s1s1 = s1 * 3   # aa: 0 OR bb: 3
        def _crunch_1c_tdm2 (d2_ijkk, i0, i1, j0, j1, k0, k1):
            d2[(s12l,s12h), i0:i1, j0:j1, k0:k1, k0:k1] = d2_ijkk
            d2[(s21l,s21h), k0:k1, k0:k1, i0:i1, j0:j1] = d2_ijkk.transpose (0,3,4,1,2)
            d2[s1s1, i0:i1, k0:k1, k0:k1, j0:j1] = -d2_ijkk[s1,...].transpose (0,3,2,1)
            d2[s1s1, k0:k1, j0:j1, i0:i1, k0:k1] = -d2_ijkk[s1,...].transpose (2,1,0,3)
        # pph (transpose from Dirac order to Mulliken order)
        d2_ijii = fac * np.multiply.outer (self.ints[i].get_1_pph (bra,ket,s1),
                                           self.ints[j].get_1_h (bra,ket,s1)).transpose (0,1,4,2,3)
        _crunch_1c_tdm2 (d2_ijii, p, q, r, s, p, q)
        # phh (transpose to bring spin to outside and then from Dirac order to Mulliken order)
        d2_ijjj = fac * np.multiply.outer (self.ints[i].get_1_p (bra,ket,s1),
                                           self.ints[j].get_1_phh (bra,ket,s1)).transpose (1,0,4,2,3)
        _crunch_1c_tdm2 (d2_ijjj, p, q, r, s, r, s)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_1c, self.dw_1c = self.dt_1c + dt, self.dw_1c + dw
        self._put_D1_(bra, ket, d1, i, j)
        self._put_D2_(bra, ket, d2, i, j)

    def _crunch_1c1d_(self, bra, ket, i, j, k, s1):
        '''Compute the reduced density matrix elements of a coupled electron-hop and
        density fluctuation.'''
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        d2 = self._get_D2_(bra, ket)
        inti, intj, intk = self.ints[i], self.ints[j], self.ints[k]
        p, q = self.get_range (i)
        r, s = self.get_range (j)
        t, u = self.get_range (k)
        fac = 1
        nelec_f_bra = self.nelec_rf[self.rootaddr[bra]]
        nelec_f_ket = self.nelec_rf[self.rootaddr[ket]]
        fac *= fermion_des_shuffle (nelec_f_bra, (i, j, k), i)
        fac *= fermion_des_shuffle (nelec_f_ket, (i, j, k), j)
        s12l = s1 * 2   # aa: 0 OR ba: 2
        s12h = s12l + 1 # ab: 1 OR bb: 3 
        s21l = s1       # aa: 0 OR ab: 1
        s21h = s21l + 2 # ba: 2 OR bb: 3
        s1s1 = s1 * 3   # aa: 0 OR bb: 3
        def _crunch_1c_tdm2 (d2_ijkk, i0, i1, j0, j1, k0, k1):
            d2[(s12l,s12h), i0:i1, j0:j1, k0:k1, k0:k1] = d2_ijkk
            d2[(s21l,s21h), k0:k1, k0:k1, i0:i1, j0:j1] = d2_ijkk.transpose (0,3,4,1,2)
            d2[s1s1, i0:i1, k0:k1, k0:k1, j0:j1] = -d2_ijkk[s1,...].transpose (0,3,2,1)
            d2[s1s1, k0:k1, j0:j1, i0:i1, k0:k1] = -d2_ijkk[s1,...].transpose (2,1,0,3)
        d1_ij = np.multiply.outer (self.ints[i].get_1_p (bra, ket, s1),
                                   self.ints[j].get_1_h (bra, ket, s1))
        d1_skk = self.ints[k].get_1_dm1 (bra, ket)
        d2_ijkk = fac * np.multiply.outer (d1_ij, d1_skk).transpose (2,0,1,3,4)
        _crunch_1c_tdm2 (d2_ijkk, p, q, r, s, t, u)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_1c1d, self.dw_1c1d = self.dt_1c1d + dt, self.dw_1c1d + dw
        self._put_D2_(bra, ket, d2, i, j, k)

    def _crunch_1s_(self, bra, ket, i, j):
        '''Compute the reduced density matrix elements of a spin unit hop; i.e.,

        <bra|i'(a)j'(b)i(b)j(a)|ket>

        i.e.,

        j ---a---> i
        i ---b---> j

        and conjugate transpose
        '''
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        d2 = self._get_D2_(bra, ket) # aa, ab, ba, bb -> 0, 1, 2, 3
        p, q = self.get_range (i)
        r, s = self.get_range (j)
        fac = -1
        d2_spsm = fac * np.multiply.outer (self.ints[i].get_1_sp (bra, ket),
                                           self.ints[j].get_1_sm (bra, ket))
        d2[1,p:q,r:s,r:s,p:q] = d2_spsm.transpose (0,3,2,1)
        d2[2,r:s,p:q,p:q,r:s] = d2_spsm.transpose (2,1,0,3)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_1s, self.dw_1s = self.dt_1s + dt, self.dw_1s + dw
        self._put_D2_(bra, ket, d2, i, j)

    def _crunch_1s1c_(self, bra, ket, i, j, k):
        '''Compute the reduced density matrix elements of a spin-charge unit hop; i.e.,

        <bra|i'(a)k'(b)j(b)k(a)|ket>

        i.e.,

        k ---a---> i
        j ---b---> k

        and conjugate transpose
        '''
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        d2 = self._get_D2_(bra, ket) # aa, ab, ba, bb -> 0, 1, 2, 3
        p, q = self.get_range (i)
        r, s = self.get_range (j)
        t, u = self.get_range (k)
        nelec_f_bra = self.nelec_rf[self.rootaddr[bra]]
        nelec_f_ket = self.nelec_rf[self.rootaddr[ket]]
        fac = -1 # a'bb'a -> a'ab'b sign
        fac *= fermion_des_shuffle (nelec_f_bra, (i, j, k), i)
        fac *= fermion_des_shuffle (nelec_f_ket, (i, j, k), j)
        sp = np.multiply.outer (self.ints[i].get_1_p (bra, ket, 0), self.ints[j].get_1_h (bra, ket, 1))
        sm = self.ints[k].get_1_sm (bra, ket)
        d2_ikkj = fac * np.multiply.outer (sp, sm).transpose (0,3,2,1) # a'bb'a -> a'ab'b transpose
        d2[1,p:q,t:u,t:u,r:s] = d2_ikkj
        d2[2,t:u,r:s,p:q,t:u] = d2_ikkj.transpose (2,3,0,1)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_1s1c, self.dw_1s1c = self.dt_1s1c + dt, self.dw_1s1c + dw
        self._put_D2_(bra, ket, d2, i, j, k)

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
        nelec_f_bra = self.nelec_rf[self.rootaddr[bra]]
        nelec_f_ket = self.nelec_rf[self.rootaddr[ket]]
        d2 = self._get_D2_(bra, ket)
        fac = 1
        if i == k:
            pp = self.ints[i].get_1_pp (bra, ket, s2lt)
            if s2lt != 1: assert (np.all (np.abs (pp + pp.T)) < 1e-8), '{}'.format (
                np.amax (np.abs (pp + pp.T)))
        else:
            pp = np.multiply.outer (self.ints[i].get_1_p (bra, ket, s11),
                                    self.ints[k].get_1_p (bra, ket, s12))
            fac *= (1,-1)[int (i>k)]
            fac *= fermion_des_shuffle (nelec_f_bra, (i, j, k, l), i)
            fac *= fermion_des_shuffle (nelec_f_bra, (i, j, k, l), k)
        if j == l:
            hh = self.ints[j].get_1_hh (bra, ket, s2lt)
            if s2lt != 1: assert (np.all (np.abs (hh + hh.T)) < 1e-8), '{}'.format (
                np.amax (np.abs (hh + hh.T)))
        else:
            hh = np.multiply.outer (self.ints[l].get_1_h (bra, ket, s12),
                                    self.ints[j].get_1_h (bra, ket, s11))
            fac *= (1,-1)[int (j>l)]
            fac *= fermion_des_shuffle (nelec_f_ket, (i, j, k, l), j)
            fac *= fermion_des_shuffle (nelec_f_ket, (i, j, k, l), l)
        d2_ijkl = fac * np.multiply.outer (pp, hh).transpose (0,3,1,2) # Dirac -> Mulliken transp
        p, q = self.get_range (i)
        r, s = self.get_range (j)
        t, u = self.get_range (k) 
        v, w = self.get_range (l)
        d2[s2, p:q,r:s,t:u,v:w] = d2_ijkl
        d2[s2T,t:u,v:w,p:q,r:s] = d2_ijkl.transpose (2,3,0,1)
        if s2 == s2T: # same-spin only: exchange happens
            d2[s2,p:q,v:w,t:u,r:s] = -d2_ijkl.transpose (0,3,2,1)
            d2[s2,t:u,r:s,p:q,v:w] = -d2_ijkl.transpose (2,1,0,3)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_2c, self.dw_2c = self.dt_2c + dt, self.dw_2c + dw
        self._put_D2_(bra, ket, d2, i, j, k, l)

    def _fn_row_has_spin (self, _crunch_fn):
        return any ((i in _crunch_fn.__name__ for i in self.interaction_has_spin))

    def _crunch_env_(self, _crunch_fn, *row):
        if self._fn_row_has_spin (_crunch_fn):
            inv = row[2:-1]
        else:
            inv = row[2:]
        with lib.temporary_env (self, **self._orbrange_env_kwargs (inv)):
            self._loop_lroots_(_crunch_fn, row, inv)
            self._finalize_crunch_env_(_crunch_fn, row, inv)

    def _finalize_crunch_env_(self, _crunch_fn, row, inv): pass

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
            d1_shape = [2,] + [norb,]*2
            d1_size = np.prod (d1_shape)
            d1 = self.d1.ravel ()[:d1_size].reshape (d1_shape)
        d2_shape = [4,] + [norb,]*4
        d2_size = np.prod (d2_shape)
        d2 = self.d2.ravel ()[:d2_size].reshape (d2_shape)
        env_kwargs = {'nlas': nlas, 'd1': d1, 'd2': d2, '_orbidx': _orbidx}
        env_kwargs.update (self._orbrange_env_kwargs_orbidx (_orbidx))
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_i, self.dw_i = self.dt_i + dt, self.dw_i + dw
        return env_kwargs

    def _orbrange_env_kwargs_orbidx (self, _orbidx):
        ndest = self.norb
        nsrc = np.count_nonzero (_orbidx)
        nthreads = lib.num_threads ()
        idx = np.ix_(_orbidx,_orbidx)
        mask = np.zeros ((self.norb,self.norb), dtype=bool)
        mask[idx] = True
        actpairs = np.where (mask.ravel ())[0]
        if nsrc==ndest:
            dblk, lblk = split_contig_array (self.norb**2,nthreads)
            sblk = dblk
        else:
            sblk, dblk, lblk = get_contig_blks (mask)
        env_kwargs = {'_nsrc_c': c_int (nsrc),
                      '_d1buf_ncol': c_int (2*(nsrc**2)),
                      '_d2buf_ncol': c_int (4*(nsrc**4)),
                      '_pdest': c_arr (actpairs.astype (np.int32)),
                      '_dblk_idx': c_arr (dblk.astype (np.int32)),
                      '_sblk_idx': c_arr (sblk.astype (np.int32)),
                      '_lblk': c_arr (lblk.astype (np.int32)),
                      '_nblk': c_int (len (lblk))}
        return env_kwargs

    def _loop_lroots_(self, _crunch_fn, row, inv):
        self._prepare_spec_addr_ovlp_(row[0], row[1], *inv)
        bra_rng = self._get_addr_range (row[0], *inv)
        ket_rng = self._get_addr_range (row[1], *inv)
        lrow = [l for l in row]
        for lrow[0], lrow[1] in product (bra_rng, ket_rng):
            _crunch_fn (*lrow)

    def _crunch_all_(self):
        for row in self.exc_1d: self._crunch_env_(self._crunch_1d_, *row)
        for row in self.exc_2d: self._crunch_env_(self._crunch_2d_, *row)
        for row in self.exc_1c: self._crunch_env_(self._crunch_1c_, *row)
        for row in self.exc_1c1d: self._crunch_env_(self._crunch_1c1d_, *row)
        for row in self.exc_1s: self._crunch_env_(self._crunch_1s_, *row)
        for row in self.exc_1s1c: self._crunch_env_(self._crunch_1s1c_, *row)
        for row in self.exc_2c: self._crunch_env_(self._crunch_2c_, *row)
        self._add_transpose_()

    def _add_transpose_(self):
        self.tdm1s += self.tdm1s.conj ().transpose (1,0,2,4,3)
        self.tdm2s += self.tdm2s.conj ().transpose (1,0,2,4,3,6,5)

    def _umat_linequiv_loop_(self, *args):
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        for ifrag, inti in enumerate (self.ints):
            for iroot, umat in inti.umat_root.items ():
                self._umat_linequiv_(ifrag, iroot, umat, *args)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_u, self.dw_u = self.dt_u + dt, self.dw_u + dw

    def _umat_linequiv_(self, ifrag, iroot, umat, *args):
        self.tdm1s = umat_dot_1frag_(self.tdm1s, umat, self.lroots, ifrag, iroot, axis=0) 
        self.tdm1s = umat_dot_1frag_(self.tdm1s, umat, self.lroots, ifrag, iroot, axis=1) 
        self.tdm2s = umat_dot_1frag_(self.tdm2s, umat, self.lroots, ifrag, iroot, axis=0) 
        self.tdm2s = umat_dot_1frag_(self.tdm2s, umat, self.lroots, ifrag, iroot, axis=1) 

    def kernel (self):
        ''' Main driver method of class.

        Returns:
            stdm1s : ndarray of shape (nroots,nroots,2,ncas,ncas)
                1-body spin-separated LAS-state transition density matrices
            stdm2s : ndarray of shape (nroots,nroots,4,ncas,ncas,ncas,ncas)
                2-body spin-separated LAS-state transition density matrices
            t0 : tuple of length 2
                timestamp of entry into this function, for profiling by caller
        '''
        t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
        self.init_profiling ()
        self.tdm1s = np.zeros ([self.nstates,]*2 + [2,] + [self.norb,]*2, dtype=self.dtype)
        self.tdm2s = np.zeros ([self.nstates,]*2 + [4,] + [self.norb,]*4, dtype=self.dtype)
        self._crunch_all_()
        self._umat_linequiv_loop_()
        return self.tdm1s, self.tdm2s, t0

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
        profile += '\n' + 'Decomposing put:'
        profile += '\n' + fmt_str.format ('gsao', self.dt_g, self.dw_g)
        profile += '\n' + fmt_str.format ('putS', self.dt_s, self.dw_s)
        return profile

def make_stdm12s (las, ci, nelec_frs, **kwargs):
    ''' Build spin-separated LAS product-state 1- and 2-body transition density matrices

    Args:
        las : instance of :class:`LASCINoSymm`
        ci : list of list of ndarrays
            Contains all CI vectors
        nelec_frs : ndarray of shape (nfrags,nroots,2)
            Number of electrons of each spin in each rootspace in each
            fragment

    Returns:
        tdm1s : ndarray of shape (nroots,2,ncas,ncas,nroots)
            Contains 1-body LAS state transition density matrices
        tdm2s : ndarray of shape (nroots,2,ncas,ncas,2,ncas,ncas,nroots)
            Contains 2-body LAS state transition density matrices
    '''
    log = lib.logger.new_logger (las, las.verbose)
    nlas = las.ncas_sub
    ncas = las.ncas
    nfrags, nroots = nelec_frs.shape[:2]
    dtype = ci[0][0].dtype
    max_memory = getattr (las, 'max_memory', las.mol.max_memory)

    # Handle possible SOC
    nelec_rs = [tuple (x) for x in nelec_frs.sum (0)]
    spin_pure = len (set (nelec_rs)) == 1
    if not spin_pure: # Engage the ``spinless mapping''
        ci = ci_map2spinless (ci, nlas, nelec_frs)
        ix = spin_shuffle_idx (nlas)
        spin_shuffle_fac = [fermion_spin_shuffle (nelec_frs[:,i,0], nelec_frs[:,i,1])
                            for i in range (nroots)]
        nlas = [2*x for x in nlas]
        nelec_frs[:,:,0] += nelec_frs[:,:,1]
        nelec_frs[:,:,1] = 0
        ncas = ncas * 2

    # First pass: single-fragment intermediates
    hopping_index, ints, lroots = frag.make_ints (las, ci, nelec_frs, nlas=nlas)
    nstates = np.sum (np.prod (lroots, axis=0))

    # Memory check
    current_memory = lib.current_memory ()[0]
    required_memory = dtype.itemsize*nstates*nstates*(2*(ncas**2)+4*(ncas**4))/1e6
    if current_memory + required_memory > max_memory:
        raise MemoryError ("current: {}; required: {}; max: {}".format (
            current_memory, required_memory, max_memory))

    # Second pass: upper-triangle
    t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
    outerprod = LSTDM (ints, nlas, hopping_index, lroots, dtype=dtype,
                           max_memory=max_memory, log=log)
    if not spin_pure:
        outerprod.spin_shuffle = spin_shuffle_fac
    lib.logger.timer (las, 'LAS-state TDM12s second intermediate indexing setup', *t0)
    tdm1s, tdm2s, t0 = outerprod.kernel ()
    lib.logger.timer (las, 'LAS-state TDM12s second intermediate crunching', *t0)
    if las.verbose >= lib.logger.TIMER_LEVEL:
        lib.logger.info (las, 'LAS-state TDM12s crunching profile:\n%s', outerprod.sprint_profile ())


    # Clean up the ``spinless mapping''
    if not spin_pure:
        kx = [True,]*2
        jx = [True,]*nstates
        tdm1s = tdm1s[np.ix_(jx,jx,kx,ix,ix)]
        tdm2s = tdm2s[np.ix_(jx,jx,kx*2,ix,ix,ix,ix)]
        n = ncas = ncas // 2
        tdm2s_ = np.zeros ((nstates, nstates, 2, 2, n, n, n, n), dtype=tdm2s.dtype)
        tdm2s_[:,:,0,0,:,:,:,:] = tdm2s[:,:,0,:n,:n,:n,:n]
        tdm2s_[:,:,0,1,:,:,:,:] = tdm2s[:,:,0,:n,:n,n:,n:]
        tdm2s_[:,:,1,0,:,:,:,:] = tdm2s[:,:,0,n:,n:,:n,:n]
        tdm2s_[:,:,1,1,:,:,:,:] = tdm2s[:,:,0,n:,n:,n:,n:]
        tdm2s = tdm2s_
        if spin_pure: # Need this if you want to always do "spinless mapping" for testing
            tdm1s_ = np.zeros ((nstates, nstates, 2, n, n), dtype=tdm1s.dtype)
            tdm1s_[:,:,0,:,:] = tdm1s[:,:,0,:n,:n]
            tdm1s_[:,:,1,:,:] = tdm1s[:,:,0,n:,n:]
            tdm1s = tdm1s_


    # Put tdm1s in PySCF convention: [p,q] -> q'p
    if spin_pure: tdm1s = tdm1s.transpose (0,2,4,3,1)
    else: tdm1s = tdm1s[:,:,0,:,:].transpose (0,3,2,1)
    tdm2s = tdm2s.reshape (nstates,nstates,2,2,ncas,ncas,ncas,ncas).transpose (0,2,4,5,3,6,7,1)

    return tdm1s, tdm2s


