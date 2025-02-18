from mrh.my_pyscf.gpu import libgpu

import pyscf 
from gpu4mrh import patch_pyscf

from mrh.tests.gpu.geometry_generator import generator
from pyscf import gto, scf, tools, mcscf, lib
from mrh.my_pyscf.mcscf.lasscf_async import LASSCF
from pyscf.mcscf import avas	

lib.logger.TIMER_LEVEL=lib.logger.INFO

# this gives you more details about the system settings. 
# see 1_631g_inp_gpu_simple.py for black box option
gpu = libgpu.libgpu_create_device()

num_gpus = libgpu.libgpu_get_num_devices(gpu)
print("num_gpus= ", num_gpus)

libgpu.libgpu_dev_properties(gpu, num_gpus)

gpu_id = 0
libgpu.libgpu_set_device(gpu, gpu_id)

# ---- inside constructor 

nfrags=1
basis='6-31g'
outputfile='1_6-31g_out_gpu.log'
mol=gto.M(use_gpu=gpu, atom=generator(nfrags),basis=basis,verbose=4,output=outputfile)
#mol.max_memory = 8000

print("\nCalling scf.RHF(mol) ; mol.use_gpu= ", mol.use_gpu)
mf=scf.RHF(mol)
mf=mf.density_fit()
mf.run()

print("\nCalling avas.kernel w/ mf.mol.use_gpu= ", mf.mol.use_gpu)
#ncas,nelecas,guess_mo_coeff = avas.kernel(mf, ['C 2p'])
ncas,nelecas,guess_mo_coeff = avas.kernel(mf, ['C 2pz'])

print("\nStarting the LASSCF calculation with use_gpu= ", gpu)
las=LASSCF(mf, list((2,)*nfrags),list((2,)*nfrags), use_gpu=gpu)
#las.max_cycle_macro = 1

frag_atom_list=[list(range(1+4*nfrag,3+4*nfrag)) for nfrag in range(nfrags)]
#print(frag_atom_list)

print("\nCalling las.set_fragments_() with las.use_gpu= ", las.use_gpu)
mo_coeff=las.set_fragments_(frag_atom_list, guess_mo_coeff)

print("\nCalling las.kernel()")
las.kernel(mo_coeff)

#use this to free GPUs for further use and get statistics of GPU usage. 
libgpu.libgpu_destroy_device(gpu)
