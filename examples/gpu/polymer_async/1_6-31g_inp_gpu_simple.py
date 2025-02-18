from mrh.my_pyscf.gpu import libgpu

import pyscf 
from gpu4mrh import patch_pyscf

from mrh.tests.gpu.geometry_generator import generator
from pyscf import gto, scf, tools, mcscf, lib
from mrh.my_pyscf.mcscf.lasscf_async import LASSCF
from pyscf.mcscf import avas	

gpu = libgpu.libgpu_init()

lib.logger.TIMER_LEVEL=lib.logger.INFO

nfrags=1
basis='6-31g'
outputfile='1_6-31g_out_gpu.log'
mol=gto.M(use_gpu=gpu, atom=generator(nfrags),basis=basis,verbose=5,output=outputfile)
mf=scf.RHF(mol)
mf=mf.density_fit()
mf.run()

#ncas,nelecas,guess_mo_coeff = avas.kernel(mf, ['C 2p'])
ncas,nelecas,guess_mo_coeff = avas.kernel(mf, ['C 2pz'])

las=LASSCF(mf, list((2,)*nfrags),list((2,)*nfrags), use_gpu=gpu)

frag_atom_list=[list(range(1+4*nfrag,3+4*nfrag)) for nfrag in range(nfrags)]
#print(frag_atom_list)
mo_coeff=las.set_fragments_(frag_atom_list, guess_mo_coeff)
las.kernel(mo_coeff)

libgpu.libgpu_destroy_device(gpu)
