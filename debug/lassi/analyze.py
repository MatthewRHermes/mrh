import numpy as np
from scipy import linalg

def transpose (rdm, arr):
    if rdm=='rdm1s':
        return arr.transpose (0,2,1)
    elif rdm=='rdm2s':
        return arr.transpose (0,1,3,2,4,6,5)
    else:
        raise Exception

for nfrag in '1frag', '2frag':
    for rdm in 'rdm1s', 'rdm2s':
        o0 = np.load (rdm + '_o0_' + nfrag + '.npy')
        o1 = np.load (rdm + '_o1_' + nfrag + '.npy')
        print (nfrag, rdm, linalg.norm (o0.imag)>1e-8,
               linalg.norm (o1-o0),
               linalg.norm (o1.conj ()-o0),
               linalg.norm (transpose (rdm, o1) - o0))
