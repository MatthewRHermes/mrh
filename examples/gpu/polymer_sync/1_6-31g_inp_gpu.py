import pyscf
from gpu4pyscf import patch_pyscf

from geometry_generator import generator
from pyscf import gto, scf, tools, mcscf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from pyscf.mcscf import avas

#output = [module.__name__ for module in sys.modules.values() if module]
#output = sorted(output)
#print("List of imported modules= ", output)

from pyscf import lib
lib.logger.TIME_LEVEL = lib.logger.INFO

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
#mol=gto.M(atom=generator(nfrags),basis=basis,verbose=5)
print(mol.__dir__())
#quit()
print("\nCalling scf.RHF(mol)")
mf=scf.RHF(mol)
mf=mf.density_fit()
mf.run()

print("\nCalling avas.kernel")
ncas,nelecas,guess_mo_coeff = avas.kernel(mf, ['C 2pz'])

print("\nStarting the LASSCF calculation with use_gpu= ", gpu)
las=LASSCF(mf, list((2,)*nfrags),list((2,)*nfrags), use_gpu=gpu)

frag_atom_list=[list(range(1+4*nfrag,3+4*nfrag)) for nfrag in range(nfrags)]
#print(frag_atom_list)

print("\nCalling las.localize_init_guess() with las.use_gpu= ", las.use_gpu)
mo_coeff=las.localize_init_guess (frag_atom_list, guess_mo_coeff)

print("\nCalling las.kernel()")
las.kernel(mo_coeff)

# -- this should all be inside of LASSCF() destructor
libgpu.libgpu_destroy_device(gpu)
# -- this should all be inside of LASSCF() destructor
