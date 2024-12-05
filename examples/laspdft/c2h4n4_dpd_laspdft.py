from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf import lassi
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf.tools.molden import from_lasscf
from c2h4n4_struct import structure as struct
from mrh.my_pyscf.lassi.spaces import all_single_excitations
import numpy as np

'''
Diagonalize-Perturb-Diagonalize Approach
'''

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
las.lasci_()

las = lassi.spaces.all_single_excitations (las)
las.lasci ()

mc = mcpdft.LASSCF(las, 'tPBE')
pdftenergy = mc.compute_pdft_energy_()[2]

lsi = lassi.LASSI(las)
lsi.kernel()

# Rediagonalizing the matrix, with diagonal elements substituted by
# the MCPDFT energy
ham = np.dot (lsi.si*lsi.e_roots[None,:], lsi.si.conj ().T)
ham[np.diag_indices_from (ham)] = pdftenergy 
e_pdft, si_pdft = np.linalg.eigh (ham)

# LASSI-PDFT
mc = mcpdft.LASSI(lsi, 'tPBE')
lsipdftenergy = mc.kernel()[0]

print("Perturb-Diag. \t\t LASSI-PDFT")
for x, y in zip(e_pdft, lsipdftenergy):
    print(f"{x:.8f} \t\t {y:.8f}")

