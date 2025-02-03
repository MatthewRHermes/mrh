from gpu4pyscf import gto as mrh_gto
from pyscf import gto
from gpu4pyscf.lib.utils import patch_cpu_kernel

import types

print(f'{gto.M} monkey-patched to include use_gpu flag')

#gto.M = patch_cpu_kernel(gto.M)(mrh_gto.mole._M)
gto.M = mrh_gto.mole._M.__get__(gto.mole)
