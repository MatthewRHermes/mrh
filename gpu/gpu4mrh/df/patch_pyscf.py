from gpu4mrh.df.df_jk import _get_jk
from pyscf.df import df_jk

from gpu4mrh.lib.utils import patch_cpu_kernel

print(f'{df_jk} monkey-patched to GPU accelerated version')
df_jk.get_jk = patch_cpu_kernel(df_jk.get_jk)(_get_jk)

from pyscf.df.df import DF
from mrh.my_pyscf.gpu import eri_cache

_cpu_reset = DF.reset

def _reset(self, mol=None):
    '''DF.reset is where PySCF drops _cderi when the geometry changes (scanner
    mode: geometry optimizers, scans, finite differences). The DF object survives
    the reset, so drop its blocks from the device ERI cache (keyed on id(dfobj)).'''
    eri_cache.evict(self)
    return _cpu_reset(self, mol)

DF.reset = _reset
print(f'{DF}.reset monkey-patched to evict the GPU ERI cache')
