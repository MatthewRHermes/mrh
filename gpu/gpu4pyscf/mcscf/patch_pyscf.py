#from gpu4pyscf.mcscf.df_jk import _get_jk
#from pyscf.df import df_jk
from gpu4pyscf.mcscf import df
__init__ = df._ERIS.__init__
#from pyscf.mcscf import df as df_pyscf
from pyscf.mcscf.df import _ERIS
#from gpu4pyscf.df import df as mrh_df
#from pyscf import df

from gpu4pyscf.lib.utils import patch_cpu_kernel

#print(f'{df_jk} monkey-patched')
print(f'{__init__} monkey-patched')
#df_jk.get_jk = patch_cpu_kernel(df_jk.get_jk)(_get_jk)

_ERIS.__init__ = patch_cpu_kernel(_ERIS.__init__)(__init__)

#print(f'{df.DF.build} monkey-patched')
#df.DF.build = mrh_df._build.__get__(df)
