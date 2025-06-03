from gpu4mrh.mcpdft import otpd
get_ontop_pair_density = otpd.get_ontop_pair_density
from pyscf import mcpdft

from gpu4mrh.lib.utils import patch_cpu_kernel

print(f'{get_ontop_pair_density} inside otpd monkey-patched to GPU accelerated version')

#old_kernel=patch_cpu_kernel(old_kernel)(new_kernel)
mcpdft.otpd.get_ontop_pair_density = patch_cpu_kernel(mcpdft.otpd.get_ontop_pair_density)(get_ontop_pair_density)

