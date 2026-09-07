from pyscf import gto, scf, mcscf, mcpdft
from mrh.my_pyscf.tools.molcas2pyscf import get_mol_from_h5
from mrh.my_pyscf.tools.molcas2pyscf import get_mo_from_h5
from mrh.my_pyscf.tools.molcas2pyscf import set_openmolcas_grid
# get_mol_from_h5 and get_mo_from_h5 must be used together, because
# OpenMolcas and PySCF put the same GTO basis functions in different internal orders

mol = gto.M (atom='h2o.xyz', basis='cc-pVDZ', output='h2o_from_scratch.log', verbose=4)
mc = mcscf.CASSCF (scf.RHF (mol).run (), 6, 6).run (natorb=True)
print ("CASSCF(6,6)/cc-pVDZ energy of water from scratch:", mc.e_tot)

mol = get_mol_from_h5 ('h2o.rasscf.h5', output='h2o_from_openmolcas.log', verbose=4)
mo_coeff = get_mo_from_h5 (mol, 'h2o.rasscf.h5')
mc = mcscf.CASSCF (scf.RHF (mol).run (), 6, 6).run (mo_coeff, natorb=True)
print ("CASSCF(6,6)/cc-pVDZ energy of water from OpenMolcas orbital guess:", mc.e_tot)

# ALong with the above, if one wants to use the OpenMolcas grid then
# they would require the corresponding `GridFile` from OpenMolcas. (v26.06, commit: 4b455201fbc72197d5123208d5ff834d8e656c9f)
# In OpenMolcas, one can generate the grid file using the following input:
'''
&SEWARD
  Grid Input
    WriteGrid
  End of Grid Input
'''

mf = scf.RHF (mol).run ()

mc = mcpdft.CASCI(mf, 'tPBE', 6, 6)
mc.kernel(mo_coeff=mo_coeff)

e_pdft = mc.e_tot
e_cas = mc.e_mcscf

mc = mcpdft.CASCI(mf, 'tPBE', 6, 6)
mc = set_openmolcas_grid(mc, 'h2o.GridFile')
mc.kernel(mo_coeff=mo_coeff,)

e_pdft2 = mc.e_tot
e_cas2 = mc.e_mcscf

print("CAS energy of water from PySCF", e_cas)
print("CAS energy of water from OpenMolcas", e_cas2)

print("CAS-tPBE energy of water from PySCF grid:", e_pdft)
print("CAS-tPBE energy of water from OpenMolcas grid:", e_pdft2)

print("Difference in CAS energies:", e_cas - e_cas2)
print("Difference in PDFT energies:", e_pdft - e_pdft2)