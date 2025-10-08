from pyscf import gto, scf, mcscf
from mrh.my_pyscf.tools.molcas2pyscf import get_mol_from_h5
from mrh.my_pyscf.tools.molcas2pyscf import get_mo_from_h5
# get_mol_from_h5 and get_mo_from_h5 must be used together, because
# OpenMolcas and PySCF put the same GTO basis functions in different internal orders

mol = gto.M (atom='h2o.xyz', basis='cc-pVDZ', output='h2o_from_scratch.log', verbose=4)
mc = mcscf.CASSCF (scf.RHF (mol).run (), 6, 6).run (natorb=True)
print ("CASSCF(6,6)/cc-pVDZ energy of water from scratch:", mc.e_tot)

mol = get_mol_from_h5 ('h2o.rasscf.h5', output='h2o_from_openmolcas.log', verbose=4)
mo_coeff = get_mo_from_h5 (mol, 'h2o.rasscf.h5')
mc = mcscf.CASSCF (scf.RHF (mol).run (), 6, 6).run (mo_coeff, natorb=True)
print ("CASSCF(6,6)/cc-pVDZ energy of water from OpenMolcas orbital guess:", mc.e_tot)



