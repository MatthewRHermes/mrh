from gpu4mrh import gto as mrh_gto
from pyscf import gto
from gpu4mrh.lib.utils import patch_cpu_kernel

import types

print(f'{gto.M} monkey-patched to include use_gpu flag')

gto.M = mrh_gto.mole._M.__get__(gto.mole)
