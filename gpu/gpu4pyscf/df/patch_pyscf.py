from gpu4pyscf.df.df_jk import _get_jk
from gpu4pyscf.df import df
from pyscf.df import df_jk
from pyscf.df.df import DF
#from gpu4pyscf.df import df as mrh_df
#from pyscf import df

from gpu4pyscf.lib.utils import patch_cpu_kernel

print(f'{df_jk} monkey-patched to GPU accelerated version')
df_jk.get_jk = patch_cpu_kernel(df_jk.get_jk)(_get_jk)

ao2mo=df.DF.ao2mo
DF.ao2mo=patch_cpu_kernel(DF.ao2mo)(ao2mo)
#print('DF ao2mo patched')
#print(f'{df.DF.build} monkey-patched')
#df.DF.build = mrh_df._build.__get__(df)
