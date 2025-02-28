from gpu4mrh.df.df_jk import _get_jk
from pyscf.df import df_jk

from gpu4mrh.lib.utils import patch_cpu_kernel

print(f'{df_jk} monkey-patched to GPU accelerated version')
df_jk.get_jk = patch_cpu_kernel(df_jk.get_jk)(_get_jk)
