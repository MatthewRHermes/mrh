# Generalizations of numpy functions to manipulate the order of data in memory when an array might
# have more than 64 dimensions

import numpy as np

def transpose (a, shape=None, axes=None, order='C'):
    if shape is None: shape = a.shape
    if axes is None: axes = list (range (a.ndim)[::-1])
    shape = list (shape).copy ()
    axes = list (axes).copy ()
    a = a.flatten (order='K')
    assert (a.size==np.prod (shape))
    assert (order.upper () in ('C', 'F'))
    assert (len (shape) == len (axes))
    ndim = len (shape)
    if order.upper () == 'F':
        axes = [ndim-1-i for i in axes[::-1]]
        shape = shape[::-1]
    p = 1
    for i in axes:
        q = np.prod (shape[:i+1])
        r = shape[i]
        q = q // r
        a = a.reshape (p,q,r,-1, order='C')
        a = np.ascontiguousarray (a.transpose (0,2,1,3))
        p = p * r
        shape[i] = 1
    return np.ravel (a, order='C')


