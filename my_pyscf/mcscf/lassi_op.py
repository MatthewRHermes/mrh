import numpy as np
from pyscf.fci.direct_spin1 import _unpack_nelec

def lst_hopping_index (fciboxes, nlas, nelelas):
    ''' Build the LAS state transition hopping index

        Args:
            fciboxes: list of h1e_zipped_fcisolvers
            nlas: list of norbs for each fragment
            nelelas: list of neleca + nelecb for each fragment

        Returns:
            hopping_index: ndarray of ints of shape (nfrags, nroots, nroots, 2)
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
    nelec_fsr = np.array ([[_unpack_nelec (fcibox._get_nelec (solver, ne))
        for solver in fcibox.fcisolvers]
        for fcibox, ne in zip (fciboxes, nelelas)]).transpose (0,2,1)
    hopping_index = np.array ([[np.subtract.outer (spin, spin)
        for spin in frag] for frag in nelec_fsr]).transpose (0,2,3,1)
    symm_index = np.all (hopping_index.sum (0) == 0, axis=2)
    zerop_index = symm_index & (np.count_nonzero (hopping_index, axis=(0,3)) == 0)
    onep_index = symm_index & (np.abs (hopping_index).sum ((0,3)) == 2)
    return hopping_index, zerop_index, onep_index

class LSTDMint (object):
    ''' Sparse-memory storage for LAS-state transition density matrix 
        single-fragment intermediates. '''

    def __init__(self, nroots):
        self.nroots = nroots
        self._h = [[None for i in nroots] for j in nroots]
        self._hh = [[None for i in nroots] for j in nroots]
        self._phh = [[None for i in nroots] for j in nroots]
        self._sm = [[None for i in nroots] for j in nroots]
        self.dm1 = [[None for i in nroots] for j in nroots]
        self.dm2 = [[None for i in nroots] for j in nroots]

    def get_h (self, i, j):
        return self._h[i][j]

    def set_h (self, i, j, x):
        self._h[i][j] = x
        return x

    def get_p (self, i, j):
        return self._h[j][i].conj ()

    def set_p (self, i, j, x):
        self._h[j][i] = x.conj ()
        return x

    def get_hh (self, i, j):
        return self._hh[i][j]

    def set_hh (self, i, j, x):
        self._hh[i][j] = x
        return x

    def get_pp (self, i, j):
        return self._hh[j][i].conj ().transpose (0,2,1)

    def set_pp (self, i, j, x):
        self._hh[j][i] = x.conj ().transpose (0,2,1)
        return x

    def get_phh (self, i, j):
        return self._phh[i][j]

    def set_phh (self, i, j, x):
        self._phh[i][j] = x
        return x

    def get_pph (self, j, i):
        return self._phh[j][i].conj ().transpose (0,2,1)

    def set_pph (self, i, j, x):
        self._phh[j][i] = x.conj ().transpose (0,2,1)
        return x

    def get_sm (self, i, j):
        return self._sm[i][j]

    def set_sm (self, i, j, x):
        self._sm[i][j] = x
        return x

    def get_sp (self, i, j):
        return self._sm[j][i].conj ()

    def set_sp (self, i, j, x):
        self._sm[j][i] = x.conj ()
        return x

def make_stdm12s (mol, ci, nlas, nelelas_state, orbsym=None, wfnsym=None):

    # First pass: single-fragment intermediates

    # Second pass: do outer products

    pass






