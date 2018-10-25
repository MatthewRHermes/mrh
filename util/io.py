import numpy as np

def prettyprint_ndarray (mat, fmt='{:9.2e}'):
    mat = np.asarray (mat)
    fmt_str = ' '.join (fmt for col in range (mat.shape[1]))
    return '\n'.join (fmt_str.format (*row) for row in mat)


