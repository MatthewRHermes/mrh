from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf import lassi
from mrh.my_pyscf import mcpdft
from c2h4n4_struct import structure as struct
from mrh.my_pyscf.lassi.spaces import all_single_excitations

# Mean field calculation
mol = struct(0, 0, '6-31g')
mol.verbose = lib.logger.INFO
mol.build()
mf = scf.RHF(mol).run()

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
mc = mcpdft.LASSI(lsi, 'tPBE')
mc = mc.multi_state()
mc.kernel()

# CASCI-PDFT in las orbitals
from pyscf import mcpdft
mc_sci = mcpdft.CASCI(mf, 'tPBE', 6, 6)
mc_sci.fcisolver = csf_solver(mol, smult=1)
mc_sci.kernel(las.mo_coeff)

mc_tci = mcpdft.CASCI(mf, 'tPBE', 6, (4, 2))
mc_tci.fcisolver = csf_solver(mol, smult=3)
mc_tci.kernel(las.mo_coeff)


# CASSCF-PDFT in las orbitals
from pyscf import mcpdft
mcas_sci = mcpdft.CASSCF(mf, 'tPBE', 6, 6)
mcas_sci.fcisolver = csf_solver(mol, smult=1)
mcas_sci.kernel(las.mo_coeff)

mcas_tci = mcpdft.CASSCF(mf, 'tPBE', 6, (4, 2))
mcas_tci.fcisolver = csf_solver(mol, smult=3)
mcas_tci.kernel(las.mo_coeff)

# Results Singlet-Triplet Gap
print("\n----Results-------\n")
print("CASSCF S-T =", 27.21139*(mcas_tci.e_mcscf - mcas_sci.e_mcscf))
print("CASCI S-T =", 27.21139*(mc_tci.e_mcscf - mc_sci.e_mcscf))
print("LASSI S-T =", 27.21139*(lsi.e_roots[1]-lsi.e_roots[0]))
print("CASSCF-LPDFT S-T =", 27.21139*(mcas_tci.e_tot - mcas_sci.e_tot))
print("CASCI-LPDFT S-T =", 27.21139*(mc_tci.e_tot - mc_sci.e_tot))
print("LASSI-LPDFT S-T =", 27.21139*(mc.e_states[1]-mc.e_states[0]))

