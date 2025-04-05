from pyscf import gto, scf, mcscf
from pyscf.mcscf import avas
from mrh.my_pyscf.guessorb.ivo import get_ivo
from pyscf.tools import molden
import numpy as np
from pyscf import lib

mol = gto.M()
mol.atom='''
  H      1.2194     -0.1652      2.1600
  C      0.6825     -0.0924      1.2087
  C     -0.7075     -0.0352      1.1973
  H     -1.2644     -0.0630      2.1393
  C     -1.3898      0.0572     -0.0114
  H     -2.4836      0.1021     -0.0204
  C     -0.6824      0.0925     -1.2088
  H     -1.2194      0.1652     -2.1599
  C      0.7075      0.0352     -1.1973
  H      1.2641      0.0628     -2.1395
  C      1.3899     -0.0572      0.0114
  H      2.4836     -0.1022      0.0205
'''
mol.basis='CC-PVDZ'
mol.spin = 0
mol.verbose = 4
mol.build()

# Mean-field object
mf = scf.RHF(mol)
mf.kernel()

# Get the improved virtual orbital and update the mean-field mo_coeff and mo_energy
mo, mo_e = get_ivo(mf, mf.mo_energy, mf.mo_coeff, mf.mo_occ)

# Don't forget to update the mf object
mf.mo_energy = mo_e
mf.mo_coeff = mo
mo = avas.kernel(mf, ['C 2py'], minao=mol.basis)[2]

mc = mcscf.CASSCF(mf, 6, 6)
ecas_withivo = mc.kernel(mo)[0]

mo_coeff = avas.kernel(mf, ['C 2py'], minao=mol.basis)[2]
mc2 = mcscf.CASSCF(mf, 6, 6)
ecas = mc2.kernel(mo_coeff)[0]

# Energy Comparision: 
# Also see the difference between number of iterations:
print('{:>20s} {:12.9f}'.format('CASSCF (with IVO+AVAS)',ecas_withivo))
print('{:>20s} {:12.9f}'.format('CASSCF (with ROHF+AVAS )',ecas))
