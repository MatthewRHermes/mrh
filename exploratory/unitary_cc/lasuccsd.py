import numpy as np
from mrh.exploratory.unitary_cc import uccsd_sym1
from mrh.exploratory.citools import lasci_ominus1
from itertools import combinations, combinations_with_replacement


def gen_uccsd_op (norb, nlas, t1_s2sym=True):
    ''' Build the fragment-interaction singles and doubles UCC operator!! '''
    t1_idx = np.zeros ((norb, norb), dtype=np.bool_)
    nfrag = len (nlas)
    for ifrag, afrag in combinations (range (nfrag), 2):
        i = sum (nlas[:ifrag])
        a = sum (nlas[:afrag])
        j = i + nlas[ifrag]
        b = a + nlas[afrag]
        t1_idx[a:b,i:j] = True
    t1_idx = np.where (t1_idx)
    a_idxs, i_idxs = list (t1_idx[0]), list (t1_idx[1])
    pq = [[p, q] for p, q in zip (*np.tril_indices (norb))]
    frag_idx = np.concatenate ([[ix,]*n for ix, n in enumerate (nlas)])
    for ab, ij in combinations_with_replacement (pq, 2):
        abij = np.concatenate ([ab, ij])
        nfint = len (np.unique (frag_idx[abij]))
        if nfint > 1:
            a_idxs.append (ab)
            i_idxs.append (ij)
    uop = uccsd_sym1.FSUCCOperator (norb, a_idxs, i_idxs, s2sym=t1_s2sym)
    return uop
        
class FCISolver (lasci_ominus1.FCISolver):
    def get_uop (self, norb, nlas):
        frozen = str (getattr (self, 'frozen', None))
        t1_s2sym = getattr (self, 't1_s2sym', True)
        if frozen.upper () == 'CI':
            return uccsd_sym1.get_uccsd_op (norb, s2sym=t1_s2sym)
        return gen_uccsd_op (norb, nlas, t1_s2sym=t1_s2sym) 
        


