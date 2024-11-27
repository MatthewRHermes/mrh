import numpy as np
from pyscf.fci.addons import _unpack_nelec, cre_a, cre_b
from pyscf.fci import cistring

def add_orbital (ci0, norb, nelec, occ_a=0, occ_b=0):
    is_list = isinstance (ci0, list)
    is_tuple = isinstance (ci0, tuple)
    neleca, nelecb = _unpack_nelec (nelec)
    ia = ib = 0
    ndeta0 = ja = cistring.num_strings (norb, neleca) 
    ndetb0 = jb = cistring.num_strings (norb, nelecb) 
    ndeta1 = cistring.num_strings (norb+1, neleca+occ_a) 
    ndetb1 = cistring.num_strings (norb+1, nelecb+occ_b) 
    if occ_a: ia, ja = ndeta1-ndeta0, ndeta1
    if occ_b: ib, jb = ndetb1-ndetb0, ndetb1
    sgn = (1,-1)[occ_b * (neleca%2)]
    ci0 = np.asarray (ci0)
    ci0 = ci0.reshape (-1,ndeta0,ndetb0)
    nvecs = len (ci0)
    ci1 = np.zeros ((nvecs, ndeta1, ndetb1), dtype=ci0.dtype)
    ci1[:,ia:ja,ib:jb] = sgn * ci0[:,:,:]
    if nvecs==1: 
        ci1 = ci1[0]
    elif is_list:
        ci1 = list (ci1)
    elif is_tuple:
        ci1 = tuple (ci1)
    return ci1

def read_orbital (ci1, norb, nelec, occ_a=0, occ_b=0):
    is_list = isinstance (ci1, list)
    is_tuple = isinstance (ci1, tuple)
    neleca, nelecb = _unpack_nelec (nelec)
    ia = ib = 0
    ndeta0 = ja = cistring.num_strings (norb, neleca) 
    ndetb0 = jb = cistring.num_strings (norb, nelecb) 
    ndeta1 = cistring.num_strings (norb+1, neleca+occ_a) 
    ndetb1 = cistring.num_strings (norb+1, nelecb+occ_b) 
    if occ_a: ia, ja = ndeta1-ndeta0, ndeta1
    if occ_b: ib, jb = ndetb1-ndetb0, ndetb1
    sgn = (1,-1)[occ_b * (neleca%2)]
    ci1 = np.asarray (ci1)
    ci1 = ci1.reshape (-1,ndeta1,ndetb1)
    nvecs = len (ci1)
    ci0 = np.zeros ((nvecs, ndeta0, ndetb0), dtype=ci1.dtype)
    ci0[:,:,:] = sgn * ci1[:,ia:ja,ib:jb]
    if nvecs==1: 
        ci0 = ci0[0]
    elif is_list:
        ci0 = list (ci0)
    elif is_tuple:
        ci0 = tuple (ci0)
    return ci0




