import numpy as np
from scipy import linalg
from pyscf import lib
from pyscf.fci.direct_spin1 import trans_rdm12s, trans_rdm1s
from pyscf.fci.direct_spin1 import contract_1e, contract_2e, absorb_h1e
from pyscf.fci.direct_uhf import contract_1e as contract_1e_uhf
from pyscf.fci.addons import cre_a, cre_b, des_a, des_b
from pyscf.fci import cistring
from itertools import product, combinations
from mrh.my_pyscf.lassi.citools import get_lroots, get_rootaddr_fragaddr
from mrh.my_pyscf.lassi.op_o1.utilities import *
from pyscf import __config__

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
            screen_linequiv : logical
                Whether to compress data by aggressively identifying linearly equivalent
                rootspaces and storing the relevant unitary matrices.
    '''

    def __init__(self, ci, hopping_index, zerop_index, onep_index, norb, nroots, nelec_rs,
                 rootaddr, fragaddr, idx_frag, mask_ints, dtype=np.float64,
                 screen_linequiv=DO_SCREEN_LINEQUIV):
        # TODO: if it actually helps, cache the "linkstr" arrays
        self.ci = ci
        self.hopping_index = hopping_index
        self.zerop_index = zerop_index
        self.onep_index = onep_index
        self.norb = norb
        self.nroots = nroots
        self.dtype = dtype
        self.nelec_r = [tuple (n) for n in nelec_rs]
        self.ovlp = [[None for i in range (nroots)] for j in range (nroots)]
        self._h = [[[None for i in range (nroots)] for j in range (nroots)] for s in (0,1)]
        self._hh = [[[None for i in range (nroots)] for j in range (nroots)] for s in (-1,0,1)] 
        self._phh = [[[None for i in range (nroots)] for j in range (nroots)] for s in (0,1)]
        self._sm = [[None for i in range (nroots)] for j in range (nroots)]
        self.dm1 = [[None for i in range (nroots)] for j in range (nroots)]
        self.dm2 = [[None for i in range (nroots)] for j in range (nroots)]
        self.linkstr = [None for i in range (nroots)]
        self.linkstrl = [None for j in range (nroots)]
        for i in range (nroots):
            la = cistring.gen_linkstr_index(range(norb), nelec_rs[i][0])
            lb = cistring.gen_linkstr_index(range(norb), nelec_rs[i][1])
            self.linkstr[i] = (la,lb)
            la = cistring.gen_linkstr_index_trilidx(range(norb), nelec_rs[i][0])
            lb = cistring.gen_linkstr_index_trilidx(range(norb), nelec_rs[i][1])
            self.linkstrl[i] = (la,lb)
        self.rootaddr = rootaddr
        self.fragaddr = fragaddr
        self.idx_frag = idx_frag
        self.mask_ints = mask_ints

        # Consistent array shape
        self.ndeta_r = np.array ([cistring.num_strings (norb, nelec[0]) for nelec in self.nelec_r])
        self.ndetb_r = np.array ([cistring.num_strings (norb, nelec[1]) for nelec in self.nelec_r])
        self.ci = [c.reshape (-1,na,nb) for c, na, nb in zip (self.ci, self.ndeta_r, self.ndetb_r)]

        self.time_crunch = self._init_crunch_(screen_linequiv)

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
        ir, jr = self.unique_root[i], self.unique_root[j]
        try:
            assert (tab[ir][jr] is not None)
            return tab[ir][jr]
        except Exception as e:
            errstr = 'frag {} failure to get element {},{}'.format (self.idx_frag, ir, jr)
            errstr = errstr + '\nhopping_index entry: {}'.format (self.hopping_index[:,ir,jr])
            raise RuntimeError (errstr)

    def try_get_tdm (self, tab, s, i, j):
        ir, jr = self.unique_root[i], self.unique_root[j]
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

    def get_1_ovlp (self, i, j):
        return self.try_get_1 (self.ovlp, i, j)

    # 1-particle 1-operator intermediate

    def get_h (self, i, j, s):
        return self.try_get (self._h, s, i, j)

    def set_h (self, i, j, s, x):
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
        x = self.setmanip (x)
        self.dm2[i][j] = x

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

        nroots, norb = self.nroots, self.norb
        t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())

        lroots = [c.shape[0] for c in ci]

        # index down to only the unique rootspaces
        self.root_unique = np.ones (self.nroots, dtype=bool)
        self.unique_root = np.arange (self.nroots, dtype=int)
        self.umat_root = {}
        for i, j in combinations (range (self.nroots), 2):
            if not self.root_unique[i]: continue
            if not self.root_unique[j]: continue
            if self.nelec_r[i] != self.nelec_r[j]: continue
            if lroots[i] != lroots[j]: continue
            if ci[i].shape != ci[j].shape: continue
            isequal = False
            if ci[i] is ci[j]: isequal = True
            elif np.all (ci[i]==ci[j]): isequal = True
            elif np.all (np.abs (ci[i]-ci[j]) < SCREEN_THRESH): isequal=True
            else:
                ci_i = ci[i].reshape (lroots[i],-1)
                ci_j = ci[j].reshape (lroots[j],-1)
                ovlp = ci_i.conj () @ ci_j.T
                isequal = np.all (np.abs (ovlp - np.eye (lroots[i])) < SCREEN_THRESH)
                                  # need extremely high precision on this one
                if screen_linequiv and (not isequal):
                    err1 = abs ((np.trace (ovlp @ ovlp.conj ().T) / lroots[i]) - 1.0)
                    err2 = abs ((np.trace (ovlp.conj ().T @ ovlp) / lroots[i]) - 1.0)
                    isequal = (err1 < SCREEN_THRESH) and (err2 < SCREEN_THRESH)
                    if isequal: 
                        u, svals, vh = linalg.svd (ovlp)
                        assert (len (svals) == lroots[i])
                        self.umat_root[j] = u @ vh
            if isequal:
                self.root_unique[j] = False
                self.unique_root[j] = i
                self.onep_index[i] |= self.onep_index[j]
                self.onep_index[:,i] |= self.onep_index[:,j]
                self.zerop_index[i] |= self.zerop_index[j]
                self.zerop_index[:,i] |= self.zerop_index[:,j]
        for i in range (self.nroots):
            assert (self.root_unique[self.unique_root[i]])
        idx_uniq = self.root_unique

        # Overlap matrix
        offs = np.cumsum (lroots)
        for i, j in combinations (np.where (idx_uniq)[0], 2):
            if self.nelec_r[i] == self.nelec_r[j]:
                ci_i = ci[i].reshape (lroots[i], -1)
                ci_j = ci[j].reshape (lroots[j], -1)
                self.ovlp[i][j] = np.dot (ci_i.conj (), ci_j.T)
                self.ovlp[j][i] = self.ovlp[i][j].conj ().T
        for i in np.where (idx_uniq)[0]:
            ci_i = ci[i].reshape (lroots[i], -1)
            self.ovlp[i][i] = np.dot (ci_i.conj (), ci_i.T)
            errmat = self.ovlp[i][i] - np.eye (lroots[i])
            if np.amax (np.abs (errmat)) > 1e-3:
                w, v = np.linalg.eigh (self.ovlp[i][i])
                errmsg = ('States w/in single Hilbert space must be orthonormal; '
                          'eigvals (ovlp) = {}')
                raise RuntimeError (errmsg.format (w))

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
            if do2:
                for i, j in product (range (bra.shape[0]), range (ket.shape[0])):
                    d1s, d2s = trans_rdm12s (bra[i], ket[j], norb, nelec,
                                             link_index=self.linkstr[iroot])
                    # Transpose based on docstring of direct_spin1.trans_rdm12s
                    tdm1s[i,j] = np.stack (d1s, axis=0).transpose (0, 2, 1)
                    tdm2s[i,j] = np.stack (d2s, axis=0)
            else:
                for i, j in product (range (bra.shape[0]), range (ket.shape[0])):
                    d1s = trans_rdm1s (bra[i], ket[j], norb, nelec,
                                       link_index=self.linkstr[iroot])
                    # Transpose based on docstring of direct_spin1.trans_rdm12s
                    tdm1s[i,j] = np.stack (d1s, axis=0).transpose (0, 2, 1)
            return tdm1s, tdm2s
        def trans_rdm1s_loop (iroot, bra, ket):
            return trans_rdm12s_loop (iroot, bra, ket, do2=False)[0]

        # Spectator fragment contribution
        spectator_index = np.all (hopping_index == 0, axis=0)
        spectator_index[~idx_uniq,:] = False
        spectator_index[:,~idx_uniq] = False
        spectator_index[np.triu_indices (nroots, k=1)] = False
        spectator_index = np.stack (np.where (spectator_index), axis=1)
        for i, j in spectator_index:
            dm1s, dm2s = trans_rdm12s_loop (j, ci[i], ci[j], do2=zerop_index[i,j])
            self.set_dm1 (i, j, dm1s)
            if zerop_index[i,j]: self.set_dm2 (i, j, dm2s)

        # Cache some b_p|i> beforehand for the sake of the spin-flip intermediate 
        # shape = (norb, lroots[ket], ndeta[ket], ndetb[*])
        hidx_ket_a = np.where (np.any (hopping_index[0] < 0, axis=0) & idx_uniq)[0]
        hidx_ket_b = np.where (np.any (hopping_index[1] < 0, axis=0) & idx_uniq)[0]
        bpvec_list = [None for ket in range (nroots)]
        for ket in hidx_ket_b:
            if np.any (np.all (hopping_index[:,:,ket] == np.array ([1,-1])[:,None], axis=0)):
                bpvec_list[ket] = np.stack ([des_b_loop (ci[ket], self.nelec_r[ket], p)
                                             for p in range (norb)], axis=0)

        # a_p|i>; shape = (norb, lroots[ket], ndeta[*], ndetb[ket])
        for ket in hidx_ket_a:
            nelec = self.nelec_r[ket]
            apket = np.stack ([des_a_loop (ci[ket], nelec, p) for p in range (norb)], axis=0)
            nelec = (nelec[0]-1, nelec[1])
            for bra in np.where ((hopping_index[0,:,ket] < 0) & idx_uniq)[0]:
                bravec = ci[bra].reshape (lroots[bra], ndeta[bra]*ndetb[bra]).conj ()
                # <j|a_p|i>
                if np.all (hopping_index[:,bra,ket] == [-1,0]):
                    self.set_h (bra, ket, 0, np.tensordot (
                        bravec, apket.reshape (norb,lroots[ket],ndeta[bra]*ndetb[bra]).T, axes=1
                    ))
                    # <j|a'_q a_r a_p|i>, <j|b'_q b_r a_p|i> - how to tell if consistent sign rule?
                    if onep_index[bra,ket]:
                        phh = np.stack ([trans_rdm1s_loop (bra, ci[bra], ketmat)
                                         for ketmat in apket], axis=-1)
                        err = np.abs (phh[:,:,0] + phh[:,:,0].transpose (0,1,2,4,3))
                        assert (np.amax (err) < 1e-8), '{}'.format (np.amax (err)) 
                        # ^ Passing this assert proves that I have the correct index
                        # and argument ordering for the call and return of trans_rdm12s
                        self.set_phh (bra, ket, 0, phh)
                # <j|b'_q a_p|i> = <j|s-|i>
                elif np.all (hopping_index[:,bra,ket] == [-1,1]):
                    bqbra = bpvec_list[bra].reshape (norb, lroots[bra], -1).conj ()
                    self.set_sm (bra, ket, np.tensordot (
                        bqbra, apket.reshape (norb, lroots[ket], -1).T, axes=1
                    ).transpose (1,2,0,3))
                # <j|b_q a_p|i>
                elif np.all (hopping_index[:,bra,ket] == [-1,-1]):
                    hh = np.zeros ((lroots[bra], lroots[ket], norb, norb), dtype = self.dtype)
                    for q, p in product (range (norb), repeat=2):
                        bq_ap_ket = des_b_loop (apket[p], nelec, q)
                        bq_ap_ket = bq_ap_ket.reshape (lroots[ket], ndeta[bra]*ndetb[bra])
                        hh[:,:,q,p] = np.dot (bravec, bq_ap_ket.T)
                    self.set_hh (bra, ket, 1, hh)
                # <j|a_q a_p|i>
                elif np.all (hopping_index[:,bra,ket] == [-2,0]):
                    hh = np.zeros ((lroots[bra], lroots[ket], norb, norb), dtype = self.dtype)
                    for q, p in combinations (range (norb), 2):
                        aq_ap_ket = des_a_loop (apket[p], nelec, q)
                        aq_ap_ket = aq_ap_ket.reshape (lroots[ket], ndeta[bra]*ndetb[bra])
                        hh[:,:,q,p] = np.dot (bravec, aq_ap_ket.T)
                    hh -= hh.transpose (0,1,3,2)
                    self.set_hh (bra, ket, 0, hh)                
                
        # b_p|i>
        for ket in hidx_ket_b:
            nelec = self.nelec_r[ket]
            bpket = np.stack ([des_b_loop (ci[ket], nelec, p)
                for p in range (norb)], axis=0) if bpvec_list[ket] is None else bpvec_list[ket]
            nelec = (nelec[0], nelec[1]-1)
            for bra in np.where ((hopping_index[1,:,ket] < 0) & idx_uniq)[0]:
                bravec = ci[bra].reshape (lroots[bra], ndeta[bra]*ndetb[bra]).conj ()
                # <j|b_p|i>
                if np.all (hopping_index[:,bra,ket] == [0,-1]):
                    self.set_h (bra, ket, 1, np.tensordot (
                        bravec, bpket.reshape (norb,lroots[ket],ndeta[bra]*ndetb[bra]).T, axes=1
                    ))
                    # <j|a'_q a_r b_p|i>, <j|b'_q b_r b_p|i> - how to tell if consistent sign rule?
                    if onep_index[bra,ket]:
                        phh = np.stack ([trans_rdm1s_loop (bra, ci[bra], ketmat)
                                         for ketmat in bpket], axis=-1)
                        err = np.abs (phh[:,:,1] + phh[:,:,1].transpose (0,1,2,4,3))
                        assert (np.amax (err) < 1e-8), '{}'.format (np.amax (err))
                        # ^ Passing this assert proves that I have the correct index
                        # and argument ordering for the call and return of trans_rdm12s
                        self.set_phh (bra, ket, 1, phh)
                # <j|b_q b_p|i>
                elif np.all (hopping_index[:,bra,ket] == [0,-2]):
                    hh = np.zeros ((lroots[bra], lroots[ket], norb, norb), dtype = self.dtype)
                    for q, p in combinations (range (norb), 2):
                        bq_bp_ket = des_b_loop (bpket[p], nelec, q)
                        bq_bp_ket = bq_bp_ket.reshape (lroots[ket], ndeta[bra]*ndetb[bra])
                        hh[:,:,q,p] = np.dot (bravec, bq_bp_ket.T)
                    hh -= hh.transpose (0,1,3,2)
                    self.set_hh (bra, ket, 2, hh)                
        
        return t0

    def contract_h00 (self, h_00, h_11, h_22, ket):
        r = self.rootaddr[ket]
        n = self.fragaddr[ket]
        norb, nelec = self.norb, self.nelec_r[r]
        ci = self.ci[r][n]
        h_uhf = (h_11[0] - h_11[1]) / 2
        h_uhf = [h_uhf, -h_uhf]
        h_11 = h_11.sum (0) / 2
        if h_22 is None:
            hci = h_00*ci + contract_1e (h_11, ci, norb, nelec,
                                         link_index=self.linkstrl[r])
        else:
            h2eff = absorb_h1e (h_11, h_22, norb, nelec, 0.5)
            hci = h_00*ci + contract_2e (h2eff, ci, norb, nelec,
                                         link_index=self.linkstrl[r])
        hci += contract_1e_uhf (h_uhf, ci, norb, nelec,
                                link_index=self.linkstrl[r])
        return hci

    def contract_h10 (self, spin, h_10, h_21, ket):
        r = self.rootaddr[ket]
        n = self.fragaddr[ket]
        norb, nelec = self.norb, self.nelec_r[r]
        cre_op = (cre_a, cre_b)[spin]
        ci = self.ci[r][n]
        hci = 0
        for p in range (self.norb):
            hci += h_10[p] * cre_op (ci, norb, nelec, p)
            if h_21 is not None:
                hci += cre_op (contract_1e (h_21[p], ci, norb, nelec),
                               norb, nelec, p)
        return hci

    def contract_h01 (self, spin, h_01, h_12, ket):
        r = self.rootaddr[ket]
        n = self.fragaddr[ket]
        norb, nelec = self.norb, self.nelec_r[r]
        des_op = (des_a, des_b)[spin]
        ci = self.ci[r][n]
        hci = 0
        nelecp = list (nelec)
        nelecp[spin] = nelecp[spin] - 1
        nelecp = tuple (nelecp)
        for p in range (self.norb):
            hci += h_01[p] * des_op (ci, norb, nelec, p)
            if h_12 is not None:
                hci += contract_1e (h_12[:,:,p], des_op (ci, norb, nelec, p),
                                    norb, nelecp)
        return hci

    def contract_h20 (self, spin, h_20, ket):
        r = self.rootaddr[ket]
        n = self.fragaddr[ket]
        norb, nelec = self.norb, self.nelec_r[r]
        ci = self.ci[r][n]
        # 0, 1, 2 = aa, ab, bb
        s11 = int (spin>1)
        s12 = int (spin>0)
        cre_op1 = (cre_a, cre_b)[s11]
        cre_op2 = (cre_a, cre_b)[s12]
        hci = 0
        nelecq = list (nelec)
        nelecq[s12] = nelecq[s12] + 1
        for q in range (self.norb):
            qci = cre_op2 (ci, norb, nelec, q)
            for p in range (self.norb):
                hci += h_20[p,q] * cre_op1 (qci, norb, nelecq, p)
        return hci

    def contract_h02 (self, spin, h_02, ket):
        r = self.rootaddr[ket]
        n = self.fragaddr[ket]
        norb, nelec = self.norb, self.nelec_r[r]
        ci = self.ci[r][n]
        # 0, 1, 2 = aa, ab, bb
        s11 = int (spin>1)
        s12 = int (spin>0)
        des_op1 = (des_a, des_b)[s11]
        des_op2 = (des_a, des_b)[s12]
        hci = 0
        nelecq = list (nelec)
        nelecq[s11] = nelecq[s11] - 1
        for q in range (self.norb):
            qci = des_op1 (ci, norb, nelec, q)
            for p in range (self.norb):
                hci += h_02[p,q] * des_op2 (qci, norb, nelecq, p)
        return hci

    def contract_h11 (self, spin, h_11, ket):
        r = self.rootaddr[ket]
        n = self.fragaddr[ket]
        norb, nelec = self.norb, self.nelec_r[r]
        ci = self.ci[r][n]
        # 0, 1, 2, 3 = aa, ab, ba, bb
        s11 = spin // 2
        s12 = spin % 2
        cre_op = (cre_a, cre_b)[s11]
        des_op = (des_a, des_b)[s12]
        hci = 0
        if nelec[s12] == 0: return hci
        nelecq = list (nelec)
        nelecq[s12] = nelecq[s12] - 1
        for q in range (self.norb):
            qci = des_op (ci, norb, nelec, q)
            for p in range (self.norb):
                hci += h_11[p,q] * cre_op (qci, norb, nelecq, p)
        return hci

def make_ints (las, ci, nelec_frs, screen_linequiv=DO_SCREEN_LINEQUIV, nlas=None,
               _FragTDMInt_class=FragTDMInt, mask_ints=None):
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

    Returns:
        hopping_index : ndarray of ints of shape (nfrags, 2, nroots, nroots)
            element [i,j,k,l] reports the change of number of electrons of
            spin j in fragment i between LAS rootspaces k and l
        ints : list of length nfrags of instances of :class:`FragTDMInt`
        lroots: ndarray of ints of shape (nfrags, nroots)
            Number of states within each fragment and rootspace
    '''
    nfrags, nroots = nelec_frs.shape[:2]
    if nlas is None: nlas = las.ncas_sub
    if mask_ints is None: mask_ints = np.ones ((nroots,nroots), dtype=bool)
    lroots = get_lroots (ci)
    hopping_index, zerop_index, onep_index = lst_hopping_index (nelec_frs)
    rootaddr, fragaddr = get_rootaddr_fragaddr (lroots)
    ints = []
    for ifrag in range (nfrags):
        tdmint = _FragTDMInt_class (ci[ifrag], hopping_index[ifrag], zerop_index, onep_index,
                                   nlas[ifrag], nroots, nelec_frs[ifrag], rootaddr,
                                   fragaddr[ifrag], ifrag, mask_ints,
                                   screen_linequiv=screen_linequiv)
        lib.logger.timer (las, 'LAS-state TDM12s fragment {} intermediate crunching'.format (
            ifrag), *tdmint.time_crunch)
        lib.logger.debug (las, 'UNIQUE ROOTSPACES OF FRAG %d: %d/%d', ifrag,
                          np.count_nonzero (tdmint.root_unique), nroots)
        ints.append (tdmint)
    return hopping_index, ints, lroots


