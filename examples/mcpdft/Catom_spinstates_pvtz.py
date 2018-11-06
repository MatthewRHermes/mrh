import sys
sys.path.append ('../../..')
from pyscf import gto, mcscf, scf, fci, dft
from mrh.my_pyscf.mcpdft import mcpdft, otfnal

''' C atom triplet-singlet gap reported in JCTC 2014, 10, 3669
    CASSCF(4,4):    1.6 eV
    tPBE:           1.1 eV
    tBLYP:          1.0 eV
    'Vertical' means triplet orbitals for singlet state
    'Relaxed' means optimized orbitals for both states
'''

mol = gto.M (atom = 'C 0 0 0', basis='cc-pvtz', spin = 2, symmetry=True)
mf = scf.RHF (mol)
mf.kernel ()
hs = mcscf.CASSCF (mf, 4, (3, 1))
ehs = hs.kernel ()[0]

mf.mo_coeff = hs.mo_coeff
ls_vert = mcscf.CASCI (mf, 4, (2,2))
ls_vert.fcisolver = fci.solver (mf.mol, singlet=True)
els_vert = ls_vert.kernel ()[0]

ls_rel = mcscf.CASSCF (mf, 4, (2,2))
ls_rel.fcisolver = fci.solver (mf.mol, singlet=True)
els_rel = ls_rel.kernel ()[0]

print ("CASSCF high-spin energy: {:.8f}".format (ehs))
print ("CASSCF (vertical) low-spin energy: {:.8f}".format (els_vert))
print ("CASSCF (relaxed) low-spin energy: {:.8f}".format (els_rel))
print ("CASSCF vertical excitation energy (eV): {:.8f}".format (27.2114 * (els_vert - ehs)))
print ("CASSCF relaxed excitation energy (eV): {:.8f}".format (27.2114 * (els_rel - ehs)))

ks = dft.UKS (mol)
ks.xc = 'pbe'
ks.grids.level = 9
ot = otfnal.transfnal (ks)

els_vert = mcpdft.kernel (ls_vert, ot)
els_rel = mcpdft.kernel (ls_rel, ot)
ehs = mcpdft.kernel (hs, ot)
print ("MC-PDFT (tPBE) high-spin energy: {:.8f}".format (ehs))
print ("MC-PDFT (tPBE) (vertical) low-spin energy: {:.8f}".format (els_vert))
print ("MC-PDFT (tPBE) (relaxed) low-spin energy: {:.8f}".format (els_rel))
print ("MC-PDFT (tPBE) vertical excitation energy (eV): {:.8f}".format (27.2114 * (els_vert - ehs)))
print ("MC-PDFT (tPBE) relaxed excitation energy (eV): {:.8f}".format (27.2114 * (els_rel - ehs)))

ks = dft.UKS (mol)
ks.xc = 'blyp'
#ks.grids.level = 9
ot = otfnal.transfnal (ks)

els_vert = mcpdft.kernel (ls_vert, ot)
els_rel = mcpdft.kernel (ls_rel, ot)
ehs = mcpdft.kernel (hs, ot)
print ("MC-PDFT (tBLYP) high-spin energy: {:.8f}".format (ehs))
print ("MC-PDFT (tBLYP) (vertical) low-spin energy: {:.8f}".format (els_vert))
print ("MC-PDFT (tBLYP) (relaxed) low-spin energy: {:.8f}".format (els_rel))
print ("MC-PDFT (tBLYP) vertical excitation energy (eV): {:.8f}".format (27.2114 * (els_vert - ehs)))
print ("MC-PDFT (tBLYP) relaxed excitation energy (eV): {:.8f}".format (27.2114 * (els_rel - ehs)))

