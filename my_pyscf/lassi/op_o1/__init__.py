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


