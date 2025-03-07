from gpu4mrh.mcscf import df
__init__ = df._ERIS.__init__
from pyscf.mcscf.df import _ERIS

from gpu4mrh.lib.utils import patch_cpu_kernel

print(f'{__init__} inside CASSCF DF ERIS monkey-patched to GPU accelerated version')

_ERIS.__init__ = patch_cpu_kernel(_ERIS.__init__)(__init__)

