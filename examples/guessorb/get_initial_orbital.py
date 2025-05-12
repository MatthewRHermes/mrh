from pyscf import gto, scf, mcscf
from mrh.my_pyscf.guessorb.guessorb import get_guessorb
from pyscf.tools import molden

mol = gto.M()
mol.atom='''
C        1.29368       -0.61373        0.00001
C       -0.00002        0.18284       -0.00008
C       -1.29349       -0.61391        0.00001
O       -0.00014        1.40152        0.00002
H        2.15162        0.06227       -0.00036
H        1.33989       -1.26654        0.88144
H        1.33960       -1.26716       -0.88097
H       -2.15150        0.06205        0.00006
H       -1.33980       -1.26701       -0.88119
H       -1.33973       -1.26701        0.88121
'''
mol.basis='CC-PVDZ'
mol.spin = 0
mol.verbose = 4
mol.build()

# Get initial orbitals. Doesn't require the mean-field wave function.
mo_energy, mo_coeff = get_guessorb(mol)

# Print these orbitals
molden.from_mo(mol, 'guessorb.molden', mo_coeff)

# Mean-field object
mf = scf.ROHF(mol)

mc = mcscf.CASSCF(mf, 4, 4)
ecas_withguessorb = mc.kernel(mo_coeff)[0]


# Now typical calculation: that is with HF orbitals
mf.kernel()

mc2 = mcscf.CASSCF(mf, 4, 4)
ecas = mc2.kernel(mf.mo_coeff)[0]

# Energy Comparision: 
# Also see the difference between number of iterations:
print('{:>20s} {:12.9f}'.format('CASSCF (with GuessOrb)',ecas_withguessorb)) # 16 Macro iterations
print('{:>20s} {:12.9f}'.format('CASSCF (with ROHFOrb )',ecas)) # 27 Macro iterations
