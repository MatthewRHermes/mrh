import pyscf 
from mrh.tests.gpu.geometry_generator import generator
from pyscf import gto, scf, tools, mcscf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from pyscf.mcscf import avas	

nfrags=1
basis='6-31g'
outputfile='1_6-31g_out.log'
mol=gto.M(atom=generator(nfrags),basis=basis,verbose=5,output=outputfile)
mf=scf.RHF(mol)
mf=mf.density_fit()
mf.run()

ncas,nelecas,guess_mo_coeff = avas.kernel(mf, ['C 2pz'])

las=LASSCF(mf, list((2,)*nfrags),list((2,)*nfrags))

frag_atom_list=[list(range(1+4*nfrag,3+4*nfrag)) for nfrag in range(nfrags)]
#print(frag_atom_list)
mo_coeff=las.localize_init_guess (frag_atom_list, guess_mo_coeff)
las.kernel(mo_coeff)
