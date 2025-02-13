# Generalizations of numpy functions to manipulate the order of data in memory when an array might
# have more than 64 dimensions

import numpy as np

def transpose (a, shape=None, axes=None, order='C'):
    '''Transpose a numpy ndarray, assuming that there might possibly be more than 64 dimensions,
    and returning as a 1d array.

    Args:
        a: ndarray of size (np.prod (shape))

    Kwargs:
        shape: sequence of length (ndim)
            The nominal current shape of the array. ndim may be larger than 64, but if so, it
            cannot correspond to a.shape (because of numpy's inherent limitations).
        axes: sequence of length (ndim)
            The target axis order
        order: 'C' or 'F'
            Indicates the context which defines the meaning of the 'shape' and 'axis' kwargs:
            either C-like, row-major order (C) or Fortran-like, column-major order (F).

    Returns:
        a: ndarray of shape (np.prod (shape),)
            A flat ndarray in which the data ordering is permuted in memory according to 'shape',
            'axes', and 'order'.
    ''' 
    if shape is None: shape = a.shape
    if axes is None: axes = list (range (a.ndim)[::-1])
    shape = list (shape).copy ()
    axes = list (axes).copy ()
    a = a.flatten (order=order)
    ndim = len (shape)
    assert (a.size==np.prod (shape)), 'inaccurate shape'
    assert (order.upper () in ('C', 'F')), 'unsupported order'
    assert (len (axes) == ndim), "shape and axes don't match"
    assert (len (list (set (axes))) == ndim), 'repeated indices'
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


