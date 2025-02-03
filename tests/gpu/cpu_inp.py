def run_cpu(nfrags,basis):
    import pyscf
    import numpy as np 
    from mrh.tests.gpu.geometry_generator import generator
    from pyscf import gto, scf, tools, mcscf, lib
    from mrh.my_pyscf.mcscf.lasscf_async import LASSCF
    from pyscf.mcscf import avas	
    lib.logger.TIMER_LEVEL=lib.logger.INFO
    outputfile=str(nfrags)+'_'+str(basis)+'_out_cpu_ref.log';
    mol=gto.M(atom=generator(nfrags),basis=basis,verbose=4,output=outputfile,max_memory=160000)
    mf=scf.RHF(mol)
    mf=mf.density_fit()
    mf.run()
    las=LASSCF(mf, list((2,)*nfrags),list((2,)*nfrags),verbose=4)#, use_gpu=gpu)
    frag_atom_list=[list(range(1+4*nfrag,3+4*nfrag)) for nfrag in range(nfrags)]
    ncas,nelecas,guess_mo_coeff=avas.kernel(mf, ["C 2pz"])
    mo_coeff=las.set_fragments_(frag_atom_list, guess_mo_coeff)
    las.kernel(mo_coeff)
    return mf, las

