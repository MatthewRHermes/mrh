from gpu4pyscf.mcscf import df
__init__ = df._ERIS.__init__
from pyscf.mcscf.df import _ERIS

from gpu4pyscf.lib.utils import patch_cpu_kernel

print(f'{__init__} monkey-patched')

_ERIS.__init__ = patch_cpu_kernel(_ERIS.__init__)(__init__)

