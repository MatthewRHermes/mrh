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

def get_rootaddr_fragaddr (lroots):
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
        for ifrag in range (nfrags):
            prods_before = np.product (lroots[:ifrag,iroot], axis=0)
            prods_after = np.product (lroots[ifrag+1:,iroot], axis=0)
            addrs = np.repeat (np.arange (lroots[ifrag,iroot]), prods_before)
            addrs = np.tile (addrs, prods_after)
            fragaddr[ifrag,i:j] = addrs
        rootaddr[i:j] = iroot
    return rootaddr, fragaddr

def umat_dot_1frag_(target, umat, lroots, ifrag, iroot, axis=0):
    '''Apply a unitary transformation for 1 fragment in 1 rootspace to a tensor
    whose target axis spans all model states.

    Args:
        target: ndarray whose length on axis 'axis' is nstates
            The object to which the unitary transformation is to be applied.
            Modified in-place.
        umat: ndarray of shape (lroots[ifrag,iroot],lroots[ifrag,iroot])
            A unitary matrix; the row axis is contracted
        lroots: ndarray of shape (nfrags, nroots)
            Number of basis states in each fragment in each rootspace
        ifrag: integer
            Fragment index for the targeted block
        iroot: integer
            Rootspace index for the targeted block

    Kwargs:
        axis: integer
            The axis of target to which umat is to be applied

    Returns:
        target: same as input target
            After application of unitary transformation'''
    nprods = np.product (lroots, axis=0)
    offs = [0,] + list (np.cumsum (nprods))
    i, j = offs[iroot], offs[iroot+1]
    newaxes = [axis,] + list (range (axis)) + list (range (axis+1, target.ndim))
    oldaxes = list (np.argsort (newaxes))
    target = target.transpose (*newaxes)
    target[i:j] = _umat_dot_1frag (target[i:j], umat, lroots[:,iroot], ifrag)
    target = target.transpose (*oldaxes)
    return target

def _umat_dot_1frag (target, umat, lroots, ifrag):
    # Remember: COLUMN-MAJOR ORDER!!
    old_shape = target.shape
    new_shape = tuple (lroots[::-1]) + old_shape[1:]
    target = target.reshape (*new_shape)
    iifrag = len (lroots) - ifrag - 1
    newaxes = [iifrag,] + list (range (iifrag)) + list (range (iifrag+1, target.ndim))
    oldaxes = list (np.argsort (newaxes))
    target = target.transpose (*newaxes)
    target = np.tensordot (umat.T, target, axes=1).transpose (*oldaxes)
    return target.reshape (*old_shape)
