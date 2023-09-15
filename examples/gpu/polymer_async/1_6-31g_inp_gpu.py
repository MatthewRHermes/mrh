import pyscf 
from gpu4pyscf import patch_pyscf

from geometry_generator import generator
from pyscf import gto, scf, tools, mcscf, lib
from mrh.my_pyscf.mcscf.lasscf_async import LASSCF
from pyscf.mcscf import avas	

#lib.logger.TIMER_LEVEL=lib.logger.INFO

# -- this should all be inside of LASSCF() constructor
from mrh.my_pyscf.gpu import libgpu
gpu = libgpu.libgpu_create_device()

num_gpus = libgpu.libgpu_get_num_devices(gpu)
print("num_gpus= ", num_gpus)

libgpu.libgpu_dev_properties(gpu, num_gpus)

gpu_id = 0
libgpu.libgpu_set_device(gpu, gpu_id)

# -- this should all be inside of LASSCF() constructor

nfrags=1
basis='6-31g'
outputfile='1_6-31g_out_gpu.log'
mol=gto.M(use_gpu=gpu, atom=generator(nfrags),basis=basis,verbose=5,output=outputfile)
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

frag_atom_list=[list(range(1+4*nfrag,3+4*nfrag)) for nfrag in range(nfrags)]
#print(frag_atom_list)

print("\nCalling las.set_fragments_() with las.use_gpu= ", las.use_gpu)
mo_coeff=las.set_fragments_(frag_atom_list, guess_mo_coeff)

print("\nCalling las.kernel()")
las.kernel(mo_coeff)

# -- this should all be inside of LASSCF() destructor?
libgpu.libgpu_destroy_device(gpu)
# -- this should all be inside of LASSCF() destructor?
