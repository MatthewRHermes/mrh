import numpy as np

def get_lroots (ci):
    '''Generate a table showing the number of states contained in a (optionally nested) list
    of CI vectors.

    Args:
        ci: list (of list of) ndarray of shape (ndeta,ndetb) or (nroots,ndeta,ndetb)
            CI vectors. The number of roots in each entry is 1 for each 1-d or 2d-ndarray and
            shape[0] for each 3d- or higher ndarray. The nesting depth equals the number of
            dimensions of the returned lroots array. An exception is raised if the sequence
            is ragged (e.g., if ci[i] is a list but ci[j] is an ndarray, or likewise at any
            depth).

    Returns:
        lroots: ndarray of integers
            Number of states stored as CI vectors in each individual ndarray within ci. The shape
            is given by the length and nesting depth of the ci lists (so a single list of ndarrays
            results in a 1d lroots; a list of lists of ndarrays results in a 2d lroots, etc.)
    '''
    lroots = []
    didrecurse = False
    dideval = False
    raggedarray = RuntimeError ("ragged argument to get_lroots")
    for c in ci:
        if isinstance (c, np.ndarray):
            if didrecurse: raise raggedarray
            dideval = True
            lroots.append (1 if c.ndim<3 else c.shape[0])
        else:
            if dideval: raise raggedarray
            didrecurse = True
            lroots.append (get_lroots (c))
    return np.asarray (lroots)

def envaddr2fragaddr (lroots):
    '''Generate an index array into a compressed fragment basis for a state in the LASSI model
    space

    Args:
        lroots: ndarray of shape (nfrags, nroots)
            Number of local roots in each fragment in each rootspace

    Returns:
        rootaddr: ndarray of shape (nstates,)
            The rootspace associated with each state
        fragaddr: ndarray of shape (nfrags, nstates)
            The ordinal designation local to each fragment of each LAS state.
    '''
    nfrags, nroots = lroots.shape
    nprods = np.product (lroots, axis=0)
    fragaddr = np.zeros ((nfrags, sum(nprods)), dtype=int)
    rootaddr = np.zeros (sum(nprods), dtype=int)
    offs = np.cumsum (nprods)
    for iroot in range (nroots):
        j = offs[iroot]
        i = j - nprods[iroot]
        addrs = np.stack (np.meshgrid (*[np.arange(l) for l in lroots[::-1,iroot]],
                                      indexing='ij'), axis=0).astype (int)
        addrs = addrs.reshape (nfrags, nprods[iroot])[::-1,:]
        fragaddr[:,i:j] = addrs
        rootaddr[i:j] = iroot
    return rootaddr, fragaddr



