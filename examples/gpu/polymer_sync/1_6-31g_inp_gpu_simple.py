from mrh.my_pyscf.gpu import libgpu

import pyscf
from gpu4mrh import patch_pyscf

from mrh.tests.gpu.geometry_generator import generator
from pyscf import gto, scf, tools, mcscf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from pyscf.mcscf import avas

from pyscf import lib

lib.logger.TIME_LEVEL = lib.logger.INFO

gpu = libgpu.libgpu_init()

nfrags=1
basis='6-31g'
outputfile='1_6-31g_out_gpu.log'
mol=gto.M(use_gpu=gpu, atom=generator(nfrags),basis=basis,verbose=4,output=outputfile)
mf=scf.RHF(mol)
mf=mf.density_fit()
mf.run()

ncas,nelecas,guess_mo_coeff = avas.kernel(mf, ['C 2pz'])

las=LASSCF(mf, list((2,)*nfrags),list((2,)*nfrags), use_gpu=gpu)

frag_atom_list=[list(range(1+4*nfrag,3+4*nfrag)) for nfrag in range(nfrags)]

mo_coeff=las.localize_init_guess (frag_atom_list, guess_mo_coeff)

las.kernel(mo_coeff)

#use this to free GPUs for further use and get statistics of GPU usage. 
libgpu.libgpu_destroy_device(gpu)
