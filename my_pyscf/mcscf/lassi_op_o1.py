import numpy as np
from pyscf import lib, fci
from pyscf.fci.direct_spin1 import _unpack_nelec
from pyscf.fci.addons import cre_a, cre_b, des_a, des_b
from itertools import product, combinations
import time

def fermion_spin_shuffle (na_list, nb_list):
    ''' Compute the sign factor corresponding to the convention
        difference between

        ... a2' a1' a0' ... b2' b1' b0' |vac>

        and

        ... a2' b2' a1' b1' a0' b0' |vac>

        where subspaces 0, 1, 2, etc. have arbitrary numbers of spin-up
        and spin-down electrons 

        Args:
            na: list of up-spin electrons for each subspace
            nb: list of down-spin electrons for each subspace

        Returns:
            sgn: +-1
    '''
    assert (len (na_list) == len (nb_list))
    nperms = 0
    for ix, nb in enumerate (nb_list[1:]):
        na = sum(na_list[:ix+1])
        nperms += na * nb
    return (1,-1)[nperms%2]

def fermion_frag_shuffle (nelec_f, frag_list):
    ''' Compute the sign factor associated with the isolation of
        particular fragments in a product of fermion field operators;
        i.e., the difference between

        ... c2' ... c1' ... c0' ... |vac>

        and

        ... c2' c1' c0' ... |vac>  

        Args:
            nelec_f: list of electron numbers per fragment for the
                whole state
            frag_list: list of fragments to coalesce

        Returns:
            sgn: +- 1
    '''

    frag_list = list (set (frag_list))
    nperms = 0
    nbtwn = 0
    for ix, frag in enumerate (frag_list[1:]):
        lfrag = frag_list[ix]
        if (frag - lfrag) > 1:
            nbtwn += sum ([nelec_f[jx] for jx in range (lfrag+1,frag)])
        if nbtwn:
            nperms += nelec_f[frag] * nbtwn
    return (1,-1)[nperms%2]

def fermion_des_shuffle (nelec_f, nfrag_idx, i):
    ''' Compute the sign factor associated with anticommuting a destruction
        operator past creation operators of unrelated fragments, i.e.,    
        
        ci ... cj' ci' ch' .. |vac> -> ... cj' ci ci' ch' ... |vac>
        
    '''
    assert (i in nfrag_idx)
    # Assuming that low orbital indices touch the vacuum first,
    # the destruction operator commutes past the high-index field
    # operators first
    nfrag_idx = list (set (nfrag_idx))[::-1]
    nelec_rel = [nelec_f[ix] for ix in nfrag_idx]
    i_rel = nfrag_idx.index (i)
    nperms = sum (nelec_rel[:i_rel]) if i_rel else 0
    return (1,-1)[nperms%2]

def lst_hopping_index (fciboxes, nlas, nelelas, idx_root):
    ''' Build the LAS state transition hopping index

        Args:
            fciboxes: list of h1e_zipped_fcisolvers
            nlas: list of norbs for each fragment
            nelelas: list of neleca + nelecb for each fragment
            idx_root: integer indices of roots in this symmetry block

        Returns:
            hopping_index: ndarray of ints of shape (nfrags, 2, nroots, nroots)
                element [i,j,k,l] reports the change of number of electrons of
                spin l in fragment i between LAS states j and k
            zerop_index: ndarray of bools of shape (nroots, nroots)
                element [i,j] is true where the ith and jth LAS states are
                connected by a null excitation; i.e., no electron, pair,
                or spin hopping or pair splitting/coalescence. This implies
                nonzero 1- and 2-body transition density matrices within
                all fragments.
            onep_index: ndarray of bools of shape (nroots, nroots)
                element [i,j] is true where the ith and jth LAS states
                are connected by exactly one electron hop from i to j or vice
                versa, implying nonzero 1-body transition density matrices
                within spectator fragments and phh/pph modes within
                source/dest fragments.
    '''
    nelelas = [sum (_unpack_nelec (ne)) for ne in nelelas]
    nelec_fsr = np.array ([[_unpack_nelec (fcibox._get_nelec (fcibox.fcisolvers[ix], ne))
        for ix in idx_root] for fcibox, ne in zip (fciboxes, nelelas)]).transpose (0,2,1)
    hopping_index = np.array ([[np.subtract.outer (spin, spin)
        for spin in frag] for frag in nelec_fsr])
    symm_index = np.all (hopping_index.sum (0) == 0, axis=0)
    zerop_index = symm_index & (np.count_nonzero (hopping_index, axis=(0,1)) == 0)
    onep_index = symm_index & (np.abs (hopping_index).sum ((0,1)) == 2)
    return hopping_index, zerop_index, onep_index

class LSTDMint1 (object):
    ''' Sparse-memory storage for LAS-state transition density matrix 
        single-fragment intermediates. '''

    def __init__(self, fcibox, norb, nelec, nroots, idx_root, dtype=np.float64):
        # I'm not sure I need linkstrl
        self.linkstrl = fcibox.states_gen_linkstr (norb, nelec, tril=True)
        self.linkstr = fcibox.states_gen_linkstr (norb, nelec, tril=False)
        self.fcisolvers = [fcibox.fcisolvers[ix] for ix in idx_root]
        self.linkstrl = [self.linkstrl[ix] for ix in idx_root]
        self.linkstr = [self.linkstr[ix] for ix in idx_root]
        self.norb = norb
        self.nelec = nelec
        self.nroots = nroots
        self.ovlp = np.zeros ((nroots, nroots), dtype=dtype)
        self.nelec_r = [_unpack_nelec (fcibox._get_nelec (solver, nelec)) for solver in self.fcisolvers]
        self._h = [[[None for i in range (nroots)] for j in range (nroots)] for s in (0,1)]
        self._hh = [[[None for i in range (nroots)] for j in range (nroots)] for s in (-1,0,1)] 
        self._phh = [[[None for i in range (nroots)] for j in range (nroots)] for s in (0,1)]
        self._sm = [[None for i in range (nroots)] for j in range (nroots)]
        self.dm1 = [[None for i in range (nroots)] for j in range (nroots)]
        self.dm2 = [[None for i in range (nroots)] for j in range (nroots)]

    # 1-particle 1-operator intermediate

    def get_h (self, i, j, s):
        return self._h[s][i][j]

    def set_h (self, i, j, s, x):
        self._h[s][i][j] = x
        return x

    def get_p (self, i, j, s):
        return self._h[s][j][i].conj ()

    # 2-particle intermediate

    def get_hh (self, i, j, s):
        return self._hh[s][i][j]

    def set_hh (self, i, j, s, x):
        self._hh[s][i][j] = x
        return x

    def get_pp (self, i, j, s):
        return self._hh[s][j][i].conj ().T

    # 1-particle 3-operator intermediate

    def get_phh (self, i, j, s):
        return self._phh[s][i][j]

    def set_phh (self, i, j, s, x):
        self._phh[s][i][j] = x
        return x

    def get_pph (self, i, j, s):
        return self._phh[s][j][i].conj ().transpose (0,3,2,1)

    # spin-hop intermediate

    def get_sm (self, i, j):
        return self._sm[i][j]

    def set_sm (self, i, j, x):
        self._sm[i][j] = x
        return x

    def get_sp (self, i, j):
        return self._sm[j][i].conj ().T

    # 1-density intermediate

    def get_dm1 (self, i, j):
        if j > i:
            return self.dm1[j][i].conj ().transpose (0, 2, 1)
        return self.dm1[i][j]

    def set_dm1 (self, i, j, x):
        if j > i:
            self.dm1[j][i] = x.conj ().transpose (0, 2, 1)
        else:
            self.dm1[i][j] = x

    # 2-density intermediate

    def get_dm2 (self, i, j):
        k, l = max (i, j), min (i, j)
        return self.dm2[k][l]

    def set_dm2 (self, i, j, x):
        if j > i:
            self.dm2[j][i] = x.conj ().transpose (0, 2, 1, 4, 3)
        else:
            self.dm2[i][j] = x

    def kernel (self, ci, hopping_index, zerop_index, onep_index):
        nroots, norb = self.nroots, self.norb
        t0 = (time.clock (), time.time ())

        # Overlap matrix
        for i, j in combinations (range (self.nroots), 2):
            if self.nelec_r[i] == self.nelec_r[j]:
                self.ovlp[i,j] = ci[i].conj ().ravel ().dot (ci[j].ravel ())
        self.ovlp += self.ovlp.T
        for i in range (self.nroots):
            self.ovlp[i,i] = ci[i].conj ().ravel ().dot (ci[i].ravel ())

        # Spectator fragment contribution
        spectator_index = np.all (hopping_index == 0, axis=0)
        spectator_index[np.triu_indices (self.nroots, k=1)] = False
        spectator_index = np.stack (np.where (spectator_index), axis=1)
        for i, j in spectator_index:
            solver = self.fcisolvers[j]
            linkstr = self.linkstr[j]
            nelec = self.nelec_r[j]
            dm1s, dm2s = solver.trans_rdm12s (ci[i], ci[j], norb, nelec, link_index=linkstr) 
            self.set_dm1 (i, j, np.stack (dm1s, axis=0).transpose (0,2,1)) # Based on docstring of direct_spin1.trans_rdm12s
            if zerop_index[i,j]: self.set_dm2 (i, j, dm2s)

        # Cache some b_p|i> beforehand for the sake of the spin-flip intermediate 
        hidx_ket_a = np.where (np.any (hopping_index[0] < 0, axis=0))[0]
        hidx_ket_b = np.where (np.any (hopping_index[1] < 0, axis=0))[0]
        bpvec_list = [None for ket in range (nroots)]
        for ket in hidx_ket_b:
            if np.any (np.all (hopping_index[:,:,ket] == np.array ([1,-1])[:,None], axis=0)):
                bpvec_list[ket] = np.stack ([des_b (ci[ket], norb, self.nelec_r[ket], p) for p in range (norb)], axis=0)

        # a_p|i>
        for ket in hidx_ket_a:
            nelec = self.nelec_r[ket]
            apket = np.stack ([des_a (ci[ket], norb, nelec, p) for p in range (norb)], axis=0)
            nelec = (nelec[0]-1, nelec[1])
            for bra in np.where (hopping_index[0,:,ket] < 0)[0]:
                bravec = ci[bra].ravel ()
                # <j|a_p|i>
                if np.all (hopping_index[:,bra,ket] == [-1,0]):
                    self.set_h (bra, ket, 0, bravec.dot (apket.reshape (norb,-1).T))
                    # <j|a'_q a_r a_p|i>, <j|b'_q b_r a_p|i> - how do I tell if I have a consistent sign rule...?
                    if onep_index[bra,ket]:
                        solver = self.fcisolvers[bra]
                        linkstr = self.linkstr[bra]
                        phh = np.stack ([solver.trans_rdm12s (ketmat, ci[bra], norb,
                            self.nelec_r[bra], link_index=linkstr)[0] for ketmat in apket],
                            axis=-1)# Arg order switched based on docstring of direct_spin1.trans_rdm12s
                        err = np.abs (phh[0] + phh[0].transpose (0,2,1))
                        assert (np.amax (err) < 1e-8), '{}'.format (np.amax (err)) 
                        # ^ Passing this assert proves that I have the correct index
                        # and argument ordering for the call and return of trans_rdm12s
                        self.set_phh (bra, ket, 0, phh)
                # <j|b'_q a_p|i> = <j|s-|i>
                elif np.all (hopping_index[:,bra,ket] == [-1,1]):
                    bqbra = bpvec_list[bra].reshape (norb, -1).conj ()
                    self.set_sm (bra, ket, np.dot (bqbra, apket.reshape (norb, -1).T))
                # <j|b_q a_p|i>
                elif np.all (hopping_index[:,bra,ket] == [-1,-1]):
                    hh = np.array ([[np.dot (bravec, des_b (pket, norb, nelec, q).ravel ())
                        for pket in apket] for q in range (norb)])
                    self.set_hh (bra, ket, 1, hh)
                # <j|a_q a_p|i>
                elif np.all (hopping_index[:,bra,ket] == [-2,0]):
                    hh_triu = [bravec.dot (des_a (apket[p], norb, nelec, q).ravel ())
                        for q, p in combinations (range (norb), 2)] 
                    hh = np.zeros ((norb, norb), dtype = apket.dtype)
                    hh[np.triu_indices (norb, k=1)] = hh_triu
                    hh -= hh.T
                    self.set_hh (bra, ket, 0, hh)                
                
        # b_p|i>
        for ket in hidx_ket_b:
            nelec = self.nelec_r[ket]
            bpket = np.stack ([des_b (ci[ket], norb, nelec, p)
                for p in range (norb)], axis=0) if bpvec_list[ket] is None else bpvec_list[ket]
            nelec = (nelec[0], nelec[1]-1)
            for bra in np.where (hopping_index[1,:,ket] < 0)[0]:
                bravec = ci[bra].ravel ()
                # <j|b_p|i>
                if np.all (hopping_index[:,bra,ket] == [0,-1]):
                    self.set_h (bra, ket, 1, bravec.dot (bpket.reshape (norb,-1).T))
                    # <j|a'_q a_r b_p|i>, <j|b'_q b_r b_p|i> - how do I tell if I have a consistent sign rule...?
                    if onep_index[bra,ket]:
                        solver = self.fcisolvers[bra]
                        linkstr = self.linkstr[bra]
                        phh = np.stack ([solver.trans_rdm12s (ketmat, ci[bra], norb,
                            self.nelec_r[bra], link_index=linkstr)[0] for ketmat in bpket],
                            axis=-1) # Arg order switched based on docstring of direct_spin1.trans_rdm12s
                        err = np.abs (phh[1] + phh[1].transpose (0,2,1))
                        assert (np.amax (err) < 1e-8), '{}'.format (np.amax (err))
                        # ^ Passing this assert proves that I have the correct index
                        # and argument ordering for the call and return of trans_rdm12s
                        self.set_phh (bra, ket, 1, phh)
                # <j|b_q b_p|i>
                elif np.all (hopping_index[:,bra,ket] == [0,-2]):
                    hh_triu = [bravec.dot (des_b (bpket[p], norb, nelec, q).ravel ())
                        for q, p in combinations (range (norb), 2)]
                    hh = np.zeros ((norb, norb), dtype = bpket.dtype)
                    hh[np.triu_indices (norb, k=1)] = hh_triu
                    hh -= hh.T
                    self.set_hh (bra, ket, 2, hh)                
        
        return t0

class LSTDMint2 (object):
    ''' Intermediate-storage convenience object for second pass of LAS-state tdm12s calculations '''
    def __init__(self, ints, nlas, hopping_index, dtype=np.float64):
        self.ints = ints
        self.nlas = nlas
        self.norb = sum (nlas)
        self.hopping_index = hopping_index
        self.nfrags, _, self.nroots, _ = hopping_index.shape
        self.dtype = dtype
        self.tdm1s = self.tdm2s = None
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

        # Provided one only looks at symmetry-allowed interactions of order 1 or 2
        findf = np.argsort ((3*hopping_index[:,0]) + hopping_index[:,1], axis=0, kind='stable')
        # The above puts the source of either charge or spin at the bottom and the destination at the top
        # Because at most 2 des/creation ops are involved, the factor of 3 sets up the order
        # a'b'ba without creating confusion between spin and charge degrees of freedom
        # The 'stable' sort keeps relative order -> sign convention!
        tril_index = np.zeros_like (conserv_index)
        tril_index[np.tril_indices (self.nroots,k=-1)] = True
        idx = conserv_index & tril_index & (nop == 0)
        self.exc_null = np.vstack (list (np.where (idx))).T
        idx = conserv_index & (nop == 2) & tril_index
        self.exc_1c = np.vstack (list (np.where (idx)) + [findf[-1][idx], findf[0][idx], ispin[idx]]).T
        idx_2e = conserv_index & (nop == 4)
        # Do splits first since splits (as opposed to coalescence) might be in triu corner
        idx = idx_2e & (ncharge_index == 3) & (np.amin (charge_index, axis=0) == -2)
        exc_split = np.vstack (list (np.where (idx)) + [findf[-1][idx], findf[0][idx], findf[-2][idx], findf[0][idx], ispin[idx]]).T
        # Also do conga-line (b: j->k ; a: k->i) in full space so that we can always use <sm> as opposed to <sp>
        idx = idx_2e & (nspin_index == 3) & (ncharge_index == 2) & (np.amin (spin_index, axis=0) == -2)
        self.exc_1s1c = np.vstack (list (np.where (idx)) + [findf[-1][idx], findf[1][idx], findf[0][idx]]).T
        # Now restrict to tril corner
        idx_2e = idx_2e & tril_index
        idx = idx_2e & (ncharge_index == 0) & (nspin_index == 2)
        self.exc_1s = np.vstack (list (np.where (idx)) + [findf[-1][idx], findf[0][idx]]).T
        idx = idx_2e & (ncharge_index == 2) & (nspin_index < 3)
        exc_pair = np.vstack (list (np.where (idx)) + [findf[-1][idx], findf[0][idx], findf[-1][idx], findf[0][idx], ispin[idx]]).T
        idx = idx_2e & (ncharge_index == 4)
        exc_scatter = np.vstack (list (np.where (idx)) + [findf[-1][idx], findf[0][idx], findf[-2][idx], findf[1][idx], ispin[idx]]).T
        # combine all two-charge interactions
        self.exc_2c = np.vstack ((exc_pair, exc_split, exc_scatter))
        # overlap tensor
        self.ovlp = np.stack ([i.ovlp for i in ints], axis=-1)
        # spin-shuffle sign vector
        self.nelec_rf = np.asarray ([[list (i.nelec_r[ket]) for i in ints] for ket in range (self.nroots)]).transpose (0,2,1)
        self.spin_shuffle = [fermion_spin_shuffle (nelec_sf[0], nelec_sf[1]) for nelec_sf in self.nelec_rf]
        self.nelec_rf = self.nelec_rf.sum (1)

    def get_range (self, i):
        p = sum (self.nlas[:i])
        q = p + self.nlas[i]
        return p, q

    def get_ovlp_fac (self, bra, ket, *inv):
        idx = np.ones (self.nfrags, dtype=np.bool_)
        idx[list (inv)] = False
        wgt = np.prod (self.ovlp[bra,ket,idx])
        uniq_frags = list (set (inv))
        wgt *= self.spin_shuffle[bra] * self.spin_shuffle[ket]
        wgt *= fermion_frag_shuffle (self.nelec_rf[bra], uniq_frags)
        wgt *= fermion_frag_shuffle (self.nelec_rf[ket], uniq_frags)
        return wgt

    # Cruncher functions
    def _crunch_null_(self, bra, ket):
        d1 = self.tdm1s[bra,ket]
        d2 = self.tdm2s[bra,ket]
        nlas = self.nlas
        for i, inti in enumerate (self.ints):
            p = sum (nlas[:i])
            q = p + nlas[i]
            d1_s_ii = inti.get_dm1 (bra, ket)
            fac = self.get_ovlp_fac (bra, ket, i)
            d1[:,p:q,p:q] = fac * np.asarray (d1_s_ii)
            d2[:,p:q,p:q,p:q,p:q] = fac * np.asarray (inti.get_dm2 (bra, ket))
            for j, intj in enumerate (self.ints[:i]):
                assert (i>j)
                r = sum (nlas[:j])
                s = r + nlas[j]
                d1_s_jj = intj.get_dm1 (bra, ket)
                d2_s_iijj = np.multiply.outer (d1_s_ii, d1_s_jj).transpose (0,3,1,2,4,5)
                d2_s_iijj = d2_s_iijj.reshape (4, q-p, q-p, s-r, s-r)
                d2_s_iijj *= self.get_ovlp_fac (bra, ket, i, j)
                d2[:,p:q,p:q,r:s,r:s] = d2_s_iijj
                d2[(0,3),r:s,r:s,p:q,p:q] = d2_s_iijj[(0,3),...].transpose (0,3,4,1,2)
                d2[(1,2),r:s,r:s,p:q,p:q] = d2_s_iijj[(2,1),...].transpose (0,3,4,1,2)
                d2[(0,3),p:q,r:s,r:s,p:q] = -d2_s_iijj[(0,3),...].transpose (0,1,4,3,2)
                d2[(0,3),r:s,p:q,p:q,r:s] = -d2_s_iijj[(0,3),...].transpose (0,3,2,1,4)

    def _crunch_1c_(self, bra, ket, i, j, s1):
        d1 = self.tdm1s[bra,ket]
        d2 = self.tdm2s[bra,ket]
        inti, intj = self.ints[i], self.ints[j]
        p, q = self.get_range (i)
        r, s = self.get_range (j)
        fac = 1
        fac = self.get_ovlp_fac (bra, ket, i, j)
        fac *= fermion_des_shuffle (self.nelec_rf[bra], (i, j), i)
        fac *= fermion_des_shuffle (self.nelec_rf[ket], (i, j), j)
        d1_ij = np.multiply.outer (self.ints[i].get_p (bra, ket, s1), self.ints[j].get_h (bra, ket, s1))
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
        # pph (transpose is from Dirac order to Mulliken order)
        d2_ijii = fac * np.multiply.outer (self.ints[i].get_pph (bra, ket, s1), self.ints[j].get_h (bra, ket, s1)).transpose (0,1,4,2,3)
        _crunch_1c_tdm2 (d2_ijii, p, q, r, s, p, q)
        # phh (transpose is to bring spin onto the outside and then from Dirac order to Mulliken order)
        d2_ijjj = fac * np.multiply.outer (self.ints[i].get_p (bra, ket, s1), self.ints[j].get_phh (bra, ket, s1)).transpose (1,0,4,2,3)
        _crunch_1c_tdm2 (d2_ijjj, p, q, r, s, r, s)
        # spectator fragment mean-field (should automatically be in Mulliken order)
        for k in range (self.nfrags):
            if k in (i, j): continue
            fac = self.get_ovlp_fac (bra, ket, i, j, k)
            fac *= fermion_des_shuffle (self.nelec_rf[bra], (i, j, k), i)
            fac *= fermion_des_shuffle (self.nelec_rf[ket], (i, j, k), j)
            t, u = self.get_range (k)
            d1_skk = self.ints[k].get_dm1 (bra, ket)
            d2_ijkk = fac * np.multiply.outer (d1_ij, d1_skk).transpose (2,0,1,3,4)
            _crunch_1c_tdm2 (d2_ijkk, p, q, r, s, t, u)

    def _crunch_1s_(self, bra, ket, i, j):
        d2 = self.tdm2s[bra, ket] # aa, ab, ba, bb -> 0, 1, 2, 3
        p, q = self.get_range (i)
        r, s = self.get_range (j)
        y, z = min (i, j), max (i, j)
        fac = -1 * self.get_ovlp_fac (bra, ket, i, j)
        d2_spsm = fac * np.multiply.outer (self.ints[i].get_sp (bra, ket), self.ints[j].get_sm (bra, ket))
        d2[1,p:q,r:s,r:s,p:q] = d2_spsm.transpose (0,3,2,1)
        d2[2,r:s,p:q,p:q,r:s] = d2_spsm.transpose (2,1,0,3)

    def _crunch_1s1c_(self, bra, ket, i, j, k):
        d2 = self.tdm2s[bra, ket] # aa, ab, ba, bb -> 0, 1, 2, 3
        p, q = self.get_range (i)
        r, s = self.get_range (j)
        t, u = self.get_range (k)
        fac = -1 * self.get_ovlp_fac (bra, ket, i, j, k) # a'bb'a -> a'ab'b sign
        fac *= fermion_des_shuffle (self.nelec_rf[bra], (i, j, k), i)
        fac *= fermion_des_shuffle (self.nelec_rf[ket], (i, j, k), j)
        sp = np.multiply.outer (self.ints[i].get_p (bra, ket, 0), self.ints[j].get_h (bra, ket, 1))
        sm = self.ints[k].get_sm (bra, ket)
        d2_ikkj = fac * np.multiply.outer (sp, sm).transpose (0,3,2,1) # a'bb'a -> a'ab'b transpose
        d2[1,p:q,t:u,t:u,r:s] = d2_ikkj
        d2[2,t:u,r:s,p:q,t:u] = d2_ikkj.transpose (2,3,0,1)

    def _crunch_2c_(self, bra, ket, i, j, k, l, s2lt):
        # s2lt: 0, 1, 2 -> aa, ab, bb
        # s2: 0, 1, 2, 3 -> aa, ab, ba, bb
        s2  = (0, 1, 3)[s2lt] # aa, ab, bb
        s2T = (0, 2, 3)[s2lt] # aa, ba, bb -> when you populate the e1 <-> e2 permutation
        s11 = s2 // 2
        s12 = s2 % 2
        d2 = self.tdm2s[bra, ket]
        fac = self.get_ovlp_fac (bra, ket, i, j, k, l)
        if i == k:
            pp = self.ints[i].get_pp (bra, ket, s2lt)
            if s2lt != 1: assert (np.all (np.abs (pp + pp.T)) < 1e-8), '{}'.format (np.amax (np.abs (pp + pp.T)))
        else:
            pp = np.multiply.outer (self.ints[i].get_p (bra, ket, s11), self.ints[k].get_p (bra, ket, s12))
            fac *= (1,-1)[i>k]
            fac *= fermion_des_shuffle (self.nelec_rf[bra], (i, j, k, l), i)
            fac *= fermion_des_shuffle (self.nelec_rf[bra], (i, j, k, l), k)
        if j == l:
            hh = self.ints[j].get_hh (bra, ket, s2lt)
            if s2lt != 1: assert (np.all (np.abs (hh + hh.T)) < 1e-8), '{}'.format (np.amax (np.abs (hh + hh.T)))
        else:
            hh = np.multiply.outer (self.ints[l].get_h (bra, ket, s12), self.ints[j].get_p (bra, ket, s11))
            fac *= (1,-1)[j>l]
            fac *= fermion_des_shuffle (self.nelec_rf[ket], (i, j, k, l), j)
            fac *= fermion_des_shuffle (self.nelec_rf[ket], (i, j, k, l), l)
        d2_ijkl = fac * np.multiply.outer (pp, hh).transpose (0,3,1,2) # Dirac -> Mulliken transpose
        p, q = self.get_range (i)
        r, s = self.get_range (j)
        t, u = self.get_range (k) 
        v, w = self.get_range (l)
        d2[s2, p:q,r:s,t:u,v:w] = d2_ijkl
        d2[s2T,t:u,v:w,p:q,r:s] = d2_ijkl.transpose (2,3,0,1)
        if s2 == s2T: # same-spin only: exchange happens
            d2[s2,p:q,v:w,t:u,r:s] = -d2_ijkl.transpose (0,3,2,1)
            d2[s2,t:u,r:s,p:q,v:w] = -d2_ijkl.transpose (2,1,0,3)

    def kernel (self):
        t0 = (time.clock (), time.time ())
        self.tdm1s = np.zeros ([self.nroots,]*2 + [2,] + [self.norb,]*2, dtype=self.dtype)
        self.tdm2s = np.zeros ([self.nroots,]*2 + [4,] + [self.norb,]*4, dtype=self.dtype)
        for row in self.exc_null: self._crunch_null_(*row)
        for row in self.exc_1c: self._crunch_1c_(*row)
        for row in self.exc_1s: self._crunch_1s_(*row)
        for row in self.exc_1s1c: self._crunch_1s1c_(*row)
        for row in self.exc_2c: self._crunch_2c_(*row)
        self.tdm1s += self.tdm1s.conj ().transpose (1,0,2,4,3)
        self.tdm2s += self.tdm2s.conj ().transpose (1,0,2,4,3,6,5)
        for state in range (self.nroots): self._crunch_null_(state, state)
        return self.tdm1s, self.tdm2s, t0

def make_stdm12s (las, ci, idx_root, **kwargs):
    fciboxes = las.fciboxes
    nlas = las.ncas_sub
    nelelas = [sum (_unpack_nelec (ne)) for ne in las.nelecas_sub]
    ncas = las.ncas
    nfrags = len (fciboxes)
    nroots = np.count_nonzero (idx_root)
    nelelas_frs = [[_unpack_nelec (fcibox._get_nelec (solver, nelecas)) for solver, ix in zip (fcibox.fcisolvers, idx_root) if ix] for fcibox, nelecas in zip (las.fciboxes, las.nelecas_sub)]
    idx_root = np.where (idx_root)[0]
    nelelas_rs = [(sum ([nefrag[i][0] for nefrag in nelelas_frs]), sum ([nefrag[i][1] for nefrag in nelelas_frs])) for i in range (nroots)]

    # First pass: single-fragment intermediates
    hopping_index, zerop_index, onep_index = lst_hopping_index (fciboxes, nlas, nelelas, idx_root)
    ints = []
    for ifrag in range (nfrags):
        tdmint = LSTDMint1 (fciboxes[ifrag], nlas[ifrag], nelelas[ifrag], nroots, idx_root)
        t0 = tdmint.kernel (ci[ifrag], hopping_index[ifrag], zerop_index, onep_index)
        lib.logger.timer (las, 'LAS-state TDM12s fragment {} intermediate crunching'.format (ifrag), *t0)        
        ints.append (tdmint)


    # Second pass: upper-triangle
    t0 = (time.clock (), time.time ())
    outerprod = LSTDMint2 (ints, nlas, hopping_index, dtype=ci[0][0].dtype)
    lib.logger.timer (las, 'LAS-state TDM12s second intermediate indexing setup', *t0)        
    tdm1s, tdm2s, t0 = outerprod.kernel ()
    lib.logger.timer (las, 'LAS-state TDM12s second intermediate crunching', *t0)        

    return tdm1s.transpose (0,2,3,4,1), tdm2s.reshape (nroots, nroots, 2, 2, ncas, ncas, ncas, ncas).transpose (0,2,4,5,3,6,7,1)

