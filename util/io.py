import sys
import numpy as np
import warnings
import traceback

'''
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback
'''

def prettyprint_ndarray (mat, fmt='{:9.2e}'):
    mat = np.asarray (mat)
    fmt_str = ' '.join (fmt for col in range (mat.shape[1]))
    return '\n'.join (fmt_str.format (*row) for row in mat)


