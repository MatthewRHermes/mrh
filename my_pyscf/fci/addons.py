import numpy as np
from pyscf.fci.addons import _unpack_nelec, des_a, des_b
from pyscf.fci import cistring

def add_doubly_occupied_orbital (ci0, norb, nelec):
    '''Add a doubly-occupied orbital in the first position'''
    is_list = isinstance (ci0, list)
    is_tuple = isinstance (ci0, tuple)
    ci0 = np.asarray (ci0)
    neleca, nelecb = _unpack_nelec (nelec)
    ndeta0 = cistring.num_strings (norb, neleca)
    ndetb0 = cistring.num_strings (norb, nelecb)
    single_vector = (ci0.shape == (ndeta0,ndetb0))
    ci0 = ci0.reshape (-1,ndeta0,ndetb0)
    ndeta1 = cistring.num_strings (norb+1, neleca+1)
    ndetb1 = cistring.num_strings (norb+1, nelecb+1)
    ci1 = np.zeros ((len (ci0), ndeta1, ndetb1), dtype=ci0.dtype)
    ci1[:,:ndeta0,:ndetb0] = ci0[:,:,:]
    if neleca % 2: ci1 *= -1
    if single_vector: 
        ci1 = ci1[0]
    elif is_list:
        ci1 = list (ci1)
    elif is_tuple:
        ci1 = tuple (ci1)
    return ci1

def add_singly_occupied_orbital (ci0, norb, nelec, spin):
    is_list = isinstance (ci0, list)
    is_tuple = isinstance (ci0, tuple)
    ci0 = add_doubly_occupied_orbital (ci0)
    neleca, nelecb = _unpack_nelec (nelec)
    norb = norb + 1
    neleca = neleca + 1
    nelecb = nelecb + 1
    ndeta = cistring.num_strings (norb, neleca)
    ndetb = cistring.num_strings (norb, nelecb)
    ci0 = np.asarray (ci0)
    single_vector = (ci0.shape == (ndeta,ndetb))
    ci0 = ci0.reshape (-1,ndeta,ndetb)
    ci1 = []
    des_op = (des_b, des_a)[spin]
    for i in range (len (ci0)):
        ci1.append (des_op (ci0[i], norb, (neleca,nelecb), 0))
    if single_vector: 
        ci1 = ci1[0]
    elif is_list:
        ci1 = list (ci1)
    elif is_tuple:
        ci1 = tuple (ci1)
    return ci1


