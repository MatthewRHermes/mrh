import pyscf 
#from geometry_generator import generator
from pyscf import gto, scf, tools, mcscf,lib
from mrh.my_pyscf.mcscf.lasscf_async import LASSCF
#from mrh.my_pyscf.mcscf.lasscf_sync_o0 import LASSCF
from pyscf.mcscf import avas	
lib.logger.TIMER_LEVEL = lib.logger.INFO

nfrags=2
basis='sto-3g'
outputfile='async_1_6-31g_out.log'
#xyz=generator(nfrags)
xyz='butadiene.xyz'
mol=gto.M(atom=xyz,basis=basis,verbose=4,output=outputfile)
mf=scf.RHF(mol)
mf=mf.density_fit()
mf.run()
ncas,nelecas,guess_mo_coeff = avas.kernel(mf, ['C 2p'])
las=LASSCF(mf, list((2,)*nfrags),list((2,)*nfrags))
frag_atom_list=[list(range(1+4*nfrag,3+4*nfrag)) for nfrag in range(nfrags)]
mo_coeff=las.set_fragments_(frag_atom_list, guess_mo_coeff)
las.kernel(mo_coeff)
#mf.mo_coeff=las.mo_coeff
#mf.analyze()

