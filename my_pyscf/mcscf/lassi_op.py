import numpy as np
from pyscf import lib, fci
from pyscf.fci.direct_spin1 import _unpack_nelec
from pyscf.fci.addons import cre_a, cre_b, des_a, des_b
from itertools import product, combinations
import time

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
        return self._phh[s][j][i].conj ().transpose (0,2,1)

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
        k, l = max (i, j), min (i, j)
        return self.dm1[k][l]

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
            self.set_dm1 (i, j, dm1s)
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
                        phh = np.stack ([solver.trans_rdm12s (ci[bra], ketmat, norb,
                            self.nelec_r[bra], link_index=linkstr)[0] for ketmat in apket], axis=-1)
                        err = np.abs (phh[0] + phh[0].transpose (0,2,1))
                        assert (np.amax (err) < 1e-8), '{}'.format (np.amax (err))
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
                        phh = np.stack ([solver.trans_rdm12s (ci[bra], ketmat, norb,
                            self.nelec_r[bra], link_index=linkstr)[0] for ketmat in bpket], axis=-1)
                        err = np.abs (phh[1] + phh[1].transpose (0,2,1))
                        assert (np.amax (err) < 1e-8), '{}'.format (np.amax (err))
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
        conserv_index = np.all (hopping_index.sum (1) == 0, axis=0)
        nsop_index = np.abs (hopping_index).sum (0) # 0,0 , 2,0 , 0,2 , 2,2 , 4,0 , 0,4
        nop_index = nsop_index.sum (0) # 0, 2, 4
        nfrag_index = np.count_nonzero (np.abs (hopping_index).sum (1), axis=0) # 0-4
        ncharge_index = np.count_nonzero ((hopping_index).sum (1), axis=0) # = 0 for spin modes
        nspin_index = nsop_index[1,:,:] // 2
        # This last ^ is somewhat magical, but notice that it corresponds to the mapping
        #   2,0 ; 4,0 -> 0 -> a or aa
        #   0,2 ; 2,2 -> 1 -> b or ab
        #   0,4       -> 2 -> bb
        # Provided one only looks at symmetry-allowed interactions of order 1 or 2
        findf = np.argsort ((2*hopping_index[:,0]) + hopping_index[:,1], axis=0, kind='stable')
        # The above puts the source of either charge or spin at the bottom and the destination at the top
        # The 'stable' sort keeps relative order -> sign convention!
        # Adding 2 to the first column sorts by up-spin FIRST and down-spin SECOND
        tril_index = np.zeros_like (conserv_index)
        tril_index[np.tril_indices (self.nroots,k=-1)] = True
        idx = conserv_index & tril_index & (nop_index == 0)
        self.exc_null = np.vstack (list (np.where (idx))).T
        idx = conserv_index & (nop_index == 2) & tril_index
        self.exc_1c = np.vstack (list (np.where (idx)) + [findf[-1][idx], findf[0][idx], nspin_index[idx]]).T
        idx_2e = conserv_index & (nop_index == 4)
        # Do splits first since splits (as opposed to coalescence) might be in triu corner
        idx = idx_2e & (ncharge_index == 3) & (np.amin (hopping_index.sum (1), axis=0) == -2)
        exc_split = np.vstack (list (np.where (idx)) + [findf[-1][idx], findf[0][idx], findf[-2][idx], findf[0][idx], nspin_index[idx]]).T
        # Now restrict to tril corner
        idx_2e = idx_2e & tril_index
        idx = idx_2e & (ncharge_index == 0) & (nspin_index == 1)
        self.exc_1s = np.vstack (list (np.where (idx)) + [findf[-1][idx], findf[0][idx]]).T
        idx = idx_2e & (ncharge_index == 2)
        exc_pair = np.vstack (list (np.where (idx)) + [findf[-1][idx], findf[0][idx], findf[-1][idx], findf[0][idx], nspin_index[idx]]).T
        idx = idx_2e & (ncharge_index == 4)
        exc_scatter = np.vstack (list (np.where (idx)) + [findf[-1][idx], findf[0][idx], findf[-2][idx], findf[1][idx], nspin_index[idx]]).T
        # combine all two-charge interactions
        self.exc_2c = np.vstack ((exc_pair, exc_split, exc_scatter))
        # overlap tensor
        self.ovlp = np.stack ([i.ovlp for i in ints], axis=-1)

    def get_range (self, i):
        p = sum (self.nlas[:i])
        q = p + self.nlas[i]
        return p, q

    def get_ovlp_fac (self, bra, ket, *inv):
        idx = np.ones (self.nfrags, dtype=np.bool_)
        idx[list (inv)] = False
        return np.prod (self.ovlp[bra,ket,idx])


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
                d2[:,r:s,r:s,p:q,p:q] = d2_s_iijj.transpose (0,3,4,1,2)
                d2[(0,3),p:q,r:s,r:s,p:q] = -d2_s_iijj[(0,3),...].transpose (0,1,4,3,2)
                d2[(0,3),r:s,p:q,p:q,r:s] = -d2_s_iijj[(0,3),...].transpose (0,3,2,1,4)

    def _crunch_1c_(self, bra, ket, i, j, s1):
        d1 = self.tdm1s[bra,ket]
        d2 = self.tdm2s[bra,ket]
        inti, intj = self.ints[i], self.ints[j]
        p, q = self.get_range (i)
        r, s = self.get_range (j)
        fac = self.get_ovlp_fac (bra, ket, i, j)
        d1_ij = np.multiply.outer (self.ints[i].get_p (bra, ket, s1), self.ints[j].get_h (bra, ket, s1))
        d1[s1,:,:] = fac * d1_ij
        s1a = s1 * 2  # aa: 0, ba: 2
        s1b = s1a + 2 # ab: 1, bb: 3 (range specifier: I want [s1a, s1a + 1], which requires s1a:s1a+2 because of how Python ranges work)
        s1s1 = s1 * 3 # aa: 0, bb: 3
        def _crunch_1c_tdm2 (d2_ijkk, i0, i1, j0, j1, k0, k1):
            d2[s1a:s1b, i0:i1, j0:j1, k0:k1, k0:k1] = d2_ijkk
            d2[s1a:s1b ,k0:k1, k0:k1, i0:i1, j0:j1] = d2_ijkk.transpose (0,3,4,1,2)
            d2[s1s1, i0:i1, k0:k1, k0:k1, j0:j1] = -d2_ijkk[s1,...].transpose (0,3,2,1)
            d2[s1s1, k0:k1, j0:j1, i0:i1, k0:k1] = -d2_ijkk[s1,...].transpose (2,1,0,3)
        # pph (transpose is from Dirac order to Mulliken order)
        d2_ijii = fac * np.multiply.outer (self.ints[i].get_pph (bra, ket, s1), self.ints[j].get_h (bra, ket, s1)).transpose (0,1,4,2,3)
        _crunch_1c_tdm2 (d2_ijii, p, q, r, s, p, q)
        # phh (transpose is to bring spin onto the outside and then from Dirac order to Mulliken order)
        d2_ijjj = fac * np.multiply.outer (self.ints[i].get_p (bra, ket, s1), self.ints[j].get_phh (bra, ket, s1)).transpose (1,0,4,2,3)
        _crunch_1c_tdm2 (d2_ijjj, p, q, r, s, r, s)
        # spectator fragment mean-field (should automatically be in Mulliken order)
        for k in range (self.nroots):
            if k in (i, j): continue
            fac = self.get_ovlp_fac (bra, ket, i, j, k)
            t, u = self.get_range (k)
            d1_skk = self.ints[k].get_dm1 (bra, ket)
            d2_ijkk = fac * np.multiply.outer (d1_ij, d1_skk).transpose (2,0,1,3,4)
            _crunch_1c_tdm2 (d2_ijkk, p, q, r, s, t, u)

    def _crunch_1s_(self, bra, ket, i, j):
        d2 = self.tdm2s[bra, ket] # aa, ab, ba, bb -> 0, 1, 2, 3
        p, q = self.get_range (i)
        r, s = self.get_range (j)
        y, z = min (i, j), max (i, j)
        # TODO: generalize this. It probably applies to -all- >1-fragment TDM elements!
        #################################################################################
        nea_y_bra = self.ints[y].nelec_r[bra][0]
        nea_y_ket = self.ints[y].nelec_r[ket][0]
        neb_z_bra = self.ints[z].nelec_r[bra][1]
        neb_z_ket = self.ints[z].nelec_r[ket][1]
        sgn = 1 - 2 * ((1 + (nea_y_bra*neb_z_bra) + (nea_y_ket*neb_z_ket)) % 2)
        #################################################################################
        fac = sgn * self.get_ovlp_fac (bra, ket, i, j)
        d2_spsm = fac * np.multiply.outer (self.ints[i].get_sp (bra, ket), self.ints[j].get_sm (bra, ket))
        d2[1,p:q,r:s,r:s,p:q] = d2_spsm.transpose (0,3,2,1)
        d2[2,r:s,p:q,p:q,r:s] = d2_spsm.transpose (2,1,0,3)

    def _crunch_2c_(self, bra, ket, i, j, k, l, s2lt):
        # s2lt: 0, 1, 2 -> aa, ab, bb
        # s2: 0, 1, 2, 3 -> aa, ab, ba, bb
        s2  = (0, 1, 3)[s2lt] # aa, ab, bb
        s2T = (0, 2, 3)[s2lt] # aa, ba, bb -> when you populate the e1 <-> e2 permutation
        s11 = s2 // 2
        s12 = s2 % 2
        d2 = self.tdm2s[bra, ket]
        if i == k:
            pp = self.ints[i].get_pp (bra, ket, s2lt)
            if s2lt == 1: assert (np.all (np.abs (pp + pp.T)) < 1e-8), '{}'.format (np.amax (np.abs (pp + pp.T)))
        else:
            pp = np.multiply.outer (self.ints[i].get_p (bra, ket, s11), self.ints[k].get_p (bra, ket, s12))
        if j == l:
            hh = self.ints[j].get_hh (bra, ket, s2lt)
            if s2lt == 1: assert (np.all (np.abs (hh + hh.T)) < 1e-8), '{}'.format (np.amax (np.abs (hh + hh.T)))
        else:
            hh = np.multiply.outer (self.ints[l].get_h (bra, ket, s12), self.ints[j].get_p (bra, ket, s11))
        fac = self.get_ovlp_fac (bra, ket, i, j, k, l)
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
        lib.logger.timer (las, 'LAS-state TDM12s intermediate crunching', *t0)        
        ints.append (tdmint)


    # Second pass: upper-triangle
    t0 = (time.clock (), time.time ())
    outerprod = LSTDMint2 (ints, nlas, hopping_index, dtype=ci[0][0].dtype)
    lib.logger.timer (las, 'LAS-state TDM12s second intermediate indexing setup', *t0)        
    tdm1s, tdm2s, t0 = outerprod.kernel ()
    lib.logger.timer (las, 'LAS-state TDM12s second intermediate crunching', *t0)        

    return tdm1s.transpose (0,2,3,4,1), tdm2s.reshape (nroots, nroots, 2, 2, ncas, ncas, ncas, ncas).transpose (0,2,4,5,3,6,7,1)

