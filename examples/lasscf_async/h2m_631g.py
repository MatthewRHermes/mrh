import numpy as np
from scipy import linalg
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.mcscf import lasscf_async as asyn
from mrh.my_pyscf.mcscf import lasscf_sync_o0 as syn

xyz = '''H 0.0 0.0 0.0
         H 1.0 0.0 0.0'''
mol = gto.M (atom = xyz, basis = '6-31g', output='h2m_631g.log',
    verbose=lib.logger.DEBUG, charge=-1, spin=1)
mf = scf.RHF (mol).run ()
frag_atom_list = ((0,),(1,))

las_syn = syn.LASSCF (mf, (2,2), ((1,1),(1,0)), spin_sub=(1,2))
mo_loc = las_syn.localize_init_guess (frag_atom_list, mf.mo_coeff)
las_syn.state_average_(weights=[.5,]*2,
                       spins=[[0,1],[1,0]],
                       smults=[[1,2],[2,1]],
                       charges=[[0,0],[1,-1]])
las_syn.kernel (mo_loc)
print ("Synchronous calculation converged?", las_syn.converged)

las_asyn = asyn.LASSCF (mf, (2,2), ((1,1),(1,0)), spin_sub=(1,2))
mo_loc = las_asyn.set_fragments_(frag_atom_list, mf.mo_coeff)
las_asyn.state_average_(weights=[.5,]*2,
                        spins=[[0,1],[1,0]],
                        smults=[[1,2],[2,1]],
                        charges=[[0,0],[1,-1]])
las_asyn.kernel (mo_loc)
print ("Asynchronous calculation converged?", las_asyn.converged)

print ("Final state energies:")
print ("{:>16s} {:>16s} {:>16s}".format ("Synchronous", "Asynchronous", "Difference"))
fmt_str = "{:16.9e} {:16.9e} {:16.9e}"
for es, ea in zip (las_syn.e_states, las_asyn.e_states): print (fmt_str.format (es, ea, ea-es))
