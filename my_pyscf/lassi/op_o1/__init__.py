from mrh.my_pyscf.lassi.op_o1.stdm import make_stdm12s
from mrh.my_pyscf.lassi.op_o1.hams2ovlp import ham
from mrh.my_pyscf.lassi.op_o1.hci import contract_ham_ci
from mrh.my_pyscf.lassi.op_o1.rdm import roots_make_rdm12s, roots_trans_rdm12s, get_fdm1_maker
from mrh.my_pyscf.lassi.op_o1.hsi import gen_contract_op_si_hdiag
from mrh.my_pyscf.lassi.op_o1.utilities import *

# NOTE: PySCF has a strange convention where
# dm1[p,q] = <q'p>, but
# dm2[p,q,r,s] = <p'r'sq>
# The return values of make_stdm12s and roots_make_rdm12s observe this convention, but
# everywhere else in this module, the more sensible convention
# dm1[p,q] = <p'q>,
# dm2[p,q,r,s] = <p'r'sq>
# is used.

def get_hdiag_orth (hdiag_raw, h_op_raw, raw2orth):
    hobj_raw = h_op_raw.parent.get_neutral (verbose=0)
    hdiag_orth = np.empty (raw2orth.shape[0], dtype=hdiag_raw.dtype)
    uniq_prod_idx = raw2orth.uniq_prod_idx
    nuniq_prod = len (uniq_prod_idx)
    hdiag_orth[:nuniq_prod] = hdiag_raw[uniq_prod_idx]
    old_roots = None
    def cmp (new, old):
        if old is None: return False
        if len (new) != len (old): return False
        if np.any (new!=old): return False
        return True
    for i, (x0, roots) in enumerate (raw2orth.gen_mixed_state_vectors (_yield_roots=True)):
        if not cmp (roots, old_roots):
            hobj_subspace = hobj_raw.get_subspace (roots, verbose=0)
            h_op_subspace = hobj_subspace.get_ham_op ()
            old_roots = roots
        hdiag_orth[i+nuniq_prod] = np.dot (x0.conj (), h_op_subspace (x0))
    return hdiag_orth



