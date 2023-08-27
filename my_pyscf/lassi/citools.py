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



