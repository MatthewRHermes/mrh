import numpy as np
from pyscf.fci.addons import _unpack_nelec, cre_a, cre_b
from pyscf.fci import cistring

def add_empty_orbital (ci0, norb, nelec):
    is_list = isinstance (ci0, list)
    is_tuple = isinstance (ci0, tuple)
    ci0 = np.asarray (ci0)
    nelec = _unpack_nelec (nelec)
    ndeta0, ndetb0 = (cistring.num_strings (norb, ne) for ne in nelec)
    norb += 1
    ndeta1, ndetb1 = (cistring.num_strings (norb, ne) for ne in nelec)
    single_vector = (ci0.shape == (ndeta0,ndetb0))
    ci0 = ci0.reshape (-1,ndeta0,ndetb0)
    ci1 = np.zeros ((len (ci0), ndeta1, ndetb1), dtype=ci0.dtype)
    ci1[:,:ndeta0,:ndetb0] = ci0[:,:,:]
    if single_vector: 
        ci1 = ci1[0]
    elif is_list:
        ci1 = list (ci1)
    elif is_tuple:
        ci1 = tuple (ci1)
    return ci1

def add_singly_occupied_orbital (ci0, norb, nelec, spin):
    '''Add a singly-occupied orbital of the given spin in the last position'''
    is_list = isinstance (ci0, list)
    is_tuple = isinstance (ci0, tuple)
    nelec = _unpack_nelec (nelec)
    ndeta0, ndetb0 = (cistring.num_strings (norb, ne) for ne in nelec)
    single_vector = (np.asarray (ci0).shape == (ndeta0,ndetb0))
    ci0 = add_empty_orbital (ci0, norb, nelec)
    ndeta1, ndetb1 = (cistring.num_strings (norb+1, ne) for ne in nelec)
    ci0 = np.asarray (ci0).reshape (-1,ndeta1,ndetb1)
    ci1 = []
    cre_op = (cre_a, cre_b)[spin]
    for i in range (len (ci0)):
        ci1.append (cre_op (ci0[i], norb+1, nelec, norb))
    if single_vector: 
        ci1 = ci1[0]
    elif is_list:
        ci1 = list (ci1)
    elif is_tuple:
        ci1 = tuple (ci1)
    return ci1

def read_dummy_orbital (ci0, norb, nelec, spin=0, occ=0):
    is_list = isinstance (ci0, list)
    is_tuple = isinstance (ci0, tuple)
    neleca, nelecb = nelecad, nelecbd = _unpack_nelec (nelec)
    ndeta = cistring.num_strings (norb, neleca)
    ndetb = cistring.num_strings (norb, nelecb)
    if occ:
        if spin: nelecbd += 1
        else: nelecad += 1
    ndetad = cistring.num_strings (norb+1, nelecad)
    ndetbd = cistring.num_strings (norb+1, nelecbd)
    ci0 = np.asarray (ci0)
    single_vector = (ci0.shape == (ndetad,ndetbd))
    ci0 = ci0.reshape (-1, ndetad, ndetbd)
    ci1 = []
    for c in ci0:
        if occ and spin: ci1.append (c[:ndeta,-ndetb:])
        elif occ: ci1.append (c[-ndeta:,:ndetb])
        else: ci1.append (c[:ndeta,:ndetb])
    if single_vector: 
        ci1 = ci1[0]
    elif is_list:
        ci1 = list (ci1)
    elif is_tuple:
        ci1 = tuple (ci1)
    return ci1
            



