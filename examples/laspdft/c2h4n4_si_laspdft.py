from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.mcscf import lassi
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf.tools.molden import from_lasscf
from c2h4n4_struct import structure as struct
from mrh.my_pyscf.mcscf.lassi_states import all_single_excitations

# Mean field calculation
mol = struct(0, 0, '6-31g')
mol.output = 'c2h4n4_si_laspdft.log'
mol.verbose = lib.logger.INFO
mol.build()
mf = scf.RHF(mol).run()


# Option-1: Perfom the SA-LASSCF calculations and feed that along with level of Charge transfer to MC-PDFT to obtain the LASSI and LASSI-PDFT energy
# SA-LASSCF: For orbitals
las = LASSCF(mf, (3,3), ((2,1),(1,2)))
las = las.state_average([0.5,0.5], spins=[[1,-1],[-1,1]], smults=[[2,2],[2,2]], charges=[[0,0],[0,0]],wfnsyms=[[1,1],[1,1]])
guess_mo = las.sort_mo([16,18,22,23,24,26])
mo0 = las.localize_init_guess((list(range (5)), list(range (5,10))), guess_mo)
las.kernel(mo0)

# LASSI-PDFT
mc = mcpdft.LASSI(las, 'tPBE', (3, 3), ((2,1),(1,2)))
mc = mc.state_average([0.5,0.5], spins=[[1,-1],[-1,1]], smults=[[2,2],[2,2]], charges=[[0,0],[0,0]],wfnsyms=[[1,1],[1,1]]) # SA orbital information
mc = all_single_excitations(mc) # Level of charge transfer
mc.fcisolver.nroots = mc.nroots 
mc.kernel(mo0)
'''
# Option-2: Pass the mf object, orbital information and level of Charge transfer information
mc = mcpdft.LASSI(mf, 'tPBE', (3, 3), ((2,1),(1,2)))
guess_mo = mc.sort_mo ([16,18,22,23,24,26])
mo0 = mc.localize_init_guess((list(range (5)), list(range (5,10))), guess_mo)
mc = mc.state_average([0.5,0.5], spins=[[1,-1],[-1,1]], smults=[[2,2],[2,2]], charges=[[0,0],[0,0]],wfnsyms=[[1,1],[1,1]])
mc = all_single_excitations(mc)
mc.fcisolver.nroots = mc.nroots
mc.kernel(mo0)
'''


# CASCI in las coeff.
from pyscf import mcpdft
mycas = mcpdft.CASCI(mf, 'tPBE', 6, 6).set(fcisolver=csf_solver(mol,smult=1))
mycas.kernel(las.mo_coeff)

# Results
print("\n----Results-------\n")
#print("State",' \t',  "LASSCF Energy",'\t\t',"LASSI Energy",'\t\t', "LASSI-PDFT Energy") 
#[print(sn,'\t',x,'\t', y,'\t', z) for sn, x, y, z in zip(list(range(mc.nroots)), mc.e_mcscf, mc.e_lassi, mc.e_tot)]

print("CASCI state-0 =", mycas.e_mcscf)
print("CASCI-PDFT state-0 =", mycas.e_tot)
print("LASSI state-0 =", mc.e_lassi[0])
print("LASSI-PDFT state-0 =", mc.e_tot[0])


