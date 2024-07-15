from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf import lassi
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf.tools.molden import from_lasscf
from c2h4n4_struct import structure as struct
from mrh.my_pyscf.lassi.spaces import all_single_excitations

# Mean field calculation
mol = struct(0, 0, '6-31g')
mol.output = 'c2h4n4_si_laspdft.log'
mol.verbose = lib.logger.INFO
mol.build()
mf = scf.RHF(mol).run()

# Option: Perfom the SA-LASSCF calculations and feed that along with level of Charge transfer to MC-PDFT to obtain the LASSI and LASSI-PDFT energy

# SA-LASSCF: For orbitals
las = LASSCF(mf, (3,3), ((2,1),(1,2)))
las = las.state_average([0.5,0.5], spins=[[1,-1],[-1,1]], smults=[[2,2],[2,2]], charges=[[0,0],[0,0]],wfnsyms=[[1,1],[1,1]])
guess_mo = las.sort_mo([16,18,22,23,24,26])
mo0 = las.localize_init_guess((list(range (5)), list(range (5,10))), guess_mo)
las.kernel(mo0)

las = lassi.spaces.all_single_excitations (las)
las.lasci ()

lsi = lassi.LASSI(las)
lsi.kernel()

# LASSI-PDFT
mc = mcpdft.LASSI(lsi, 'tPBE', (3, 3), ((2,1),(1,2)), states=[0, 1])
mc.kernel() 

# CASCI-PDFT in las orbitals
from pyscf import mcpdft
mc_ci = mcpdft.CASCI(mf, 'tPBE', 6, 6)
mc_ci.kernel(las.mo_coeff)

# Results
print("\n----Results-------\n")
print("CASCI state-0 =", mc_ci.e_mcscf)
print("LASSI state-0 =", lsi.e_roots[0])

print("CASCI-PDFT state-0 =", mc_ci.e_tot)
print("LASSI-PDFT state-0 =", mc.e_tot[0])

