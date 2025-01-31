from mrh.my_pyscf.gpu import libgpu

import pyscf   # -- this is contaminating a path preventing an OpenMP runtime that supports GPUs from being picked up
from gpu4pyscf import patch_pyscf

from pyscf import gto, scf, tools, mcscf, lib
from mrh.my_pyscf.mcscf.lasscf_async import LASSCF
from pyscf.mcscf import avas	

gpu = libgpu.libgpu_create_device()

num_gpus = libgpu.libgpu_get_num_devices(gpu)
print("num_gpus= ", num_gpus)

libgpu.libgpu_dev_properties(gpu, num_gpus)

gpu_id = 0
libgpu.libgpu_set_device(gpu, gpu_id)

libgpu.libgpu_destroy_device(gpu)
