from gpu4mrh import gto as mrh_gto
from pyscf import gto

print(f'{gto.M} monkey-patched to accept the use_gpu=<device> per-Mole override')

gto.M = mrh_gto.mole._M.__get__(gto.mole)
