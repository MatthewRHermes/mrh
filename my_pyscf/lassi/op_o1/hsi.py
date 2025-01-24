from mrh.my_pyscf.lassi.op_o1.hams2ovlp import ham
from mrh.my_pyscf.lassi.citools import _fake_gen_contract_op_si_hdiag
import functools

gen_contract_op_si_hdiag = functools.partial (_fake_gen_contract_op_si_hdiag, ham)

