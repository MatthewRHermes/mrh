from mrh.tests.lasscf.c2h6n4_struct import structure as struct
from pyscf.tools import molden
from pyscf import scf, mcscf
from mrh.my_pyscf.mcscf.lasscf_sync_o0 import LASSCF

mol = struct (0.2, 0.2, '6-31g', symmetry=False)
mol.spin = 4
mol.output = 'preserving_rohf_reference.log'
mol.verbose = 4
mol.build ()
mf = scf.RHF (mol).run (max_cycle=100)
print ("Hartree--Fock energy:", mf.e_tot)

molden.from_scf (mf, 'hf.molden')

mc = mcscf.CASCI (mf, 4, 4).run ()
print ("CASCI(4,4) energy:", mc.e_tot)

# For a singly-occupied, high-spin active space, LASCI == ROHF
# (as long as you use freeze_cas_spaces=True)
las = LASSCF (mf, (2, 2), ((2,0), (2,0)), spin_sub=(3,3))
las.mo_coeff = las.localize_init_guess ([[1,2],[9,10]], mf.mo_coeff, freeze_cas_spaces=True)
las.ci = las.get_init_guess_ci ()
print ("LASCI(4,4) energy:", las.energy_elec () + las.energy_nuc ())

# But when you add doubly-occupied orbitals to the active space, the localization
# usually deoptimizes them a bit 
las = LASSCF (mf, (4, 4), ((4,2), (4,2)), spin_sub=(3,3))
las.mo_coeff = las.localize_init_guess ([[1,2],[9,10]], mf.mo_coeff, freeze_cas_spaces=True)
las.ci = las.get_init_guess_ci ()
print ("LASCI(12,8) guess energy:",
       las.energy_elec () + las.energy_nuc ())

# You can fix this by passing the MO occupancy list instead of freeze_cas_spaces=True.
# This prevents, i.e., singly- and doubly-occupied orbitals from mixing during the
# localization.
las = LASSCF (mf, (4, 4), ((4,2), (4,2)), spin_sub=(3,3))
las.mo_coeff = las.localize_init_guess ([[1,2],[9,10]], mf.mo_coeff, mo_occ=mf.mo_occ)
las.ci = las.get_init_guess_ci ()
print ("LASCI(12,8) guess energy, requiring fixed ROHF occupancies:",
       las.energy_elec () + las.energy_nuc ())


