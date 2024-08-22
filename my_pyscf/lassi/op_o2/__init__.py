from mrh.my_pyscf.lassi.op_o2.stdm import make_stdm12s
from mrh.my_pyscf.lassi.op_o2.hams2ovlp import ham
from mrh.my_pyscf.lassi.op_o2.hci import contract_ham_ci
from mrh.my_pyscf.lassi.op_o2.rdm import roots_make_rdm12s, get_fdm1_maker

# NOTE: PySCF has a strange convention where
# dm1[p,q] = <q'p>, but
# dm2[p,q,r,s] = <p'r'sq>
# The return values of make_stdm12s and roots_make_rdm12s observe this convention, but
# everywhere else in this module, the more sensible convention
# dm1[p,q] = <p'q>,
# dm2[p,q,r,s] = <p'r'sq>
# is used.


