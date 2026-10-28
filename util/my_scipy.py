from scipy.sparse import linalg as sparse_linalg

class CallbackLinearOperator (sparse_linalg.LinearOperator):
    def __init__(self, parent, shape, dtype=None, matvec=None):
        self.parent = parent
        sparse_linalg.LinearOperator.__init__(self, dtype=dtype, shape=shape)
        self._matvec_fn = matvec
                             
    def _matvec (self, x):   
        # Just to shut up the stupid warning
        return self._matvec_fn (x)

