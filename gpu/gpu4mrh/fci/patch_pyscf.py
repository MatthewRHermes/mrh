from gpu4mrh.fci.rdm import _make_rdm1_spin1, _make_rdm12_spin1, _reorder_rdm
from pyscf.fci import rdm
from gpu4mrh.fci.direct_spin1 import _trans_rdm12s, _trans_rdm1s
from pyscf.fci import direct_spin1

from gpu4mrh.lib.utils import patch_cpu_kernel

rdm.make_rdm1_spin1 = patch_cpu_kernel(rdm.make_rdm1_spin1)(_make_rdm1_spin1)
print(f'{rdm.make_rdm1_spin1} inside FCI rdm monkey-patched to GPU accelerated version', flush=True)
rdm.make_rdm12_spin1 = patch_cpu_kernel(rdm.make_rdm12_spin1)(_make_rdm12_spin1)
print(f'{rdm.make_rdm12_spin1} inside FCI rdm monkey-patched to GPU accelerated version', flush=True)
rdm.reorder_rdm = patch_cpu_kernel(rdm.reorder_rdm)(_reorder_rdm)
print(f'{rdm.reorder_rdm} inside FCI rdm monkey-patched to GPU accelerated version', flush=True)

direct_spin1.trans_rdm12s = patch_cpu_kernel(direct_spin1.trans_rdm12s)(_trans_rdm12s)
print(f'{direct_spin1.trans_rdm12s} inside FCI direct_spin1 monkey-patched to GPU accelerated version', flush=True)
direct_spin1.trans_rdm1s = patch_cpu_kernel(direct_spin1.trans_rdm1s)(_trans_rdm1s)
print(f'{direct_spin1.trans_rdm1s} inside FCI direct_spin1 monkey-patched to GPU accelerated version', flush=True)
