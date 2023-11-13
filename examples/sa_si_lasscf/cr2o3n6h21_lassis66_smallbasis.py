import numpy as np
from pyscf import gto, scf, mcscf
from pyscf.lib import chkfile
from pyscf.data import nist
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf import lassi

au2cm = nist.HARTREE2J / nist.PLANCK / nist.LIGHT_SPEED_SI * 1e-2
def yamaguchi (e_roots, s2):
    '''The Yamaguchi formula for six unpaired electrons'''
    idx = np.argsort (e_roots)
    e_roots = e_roots[idx]
    s2 = s2[idx]
    idx_hs = (np.around (s2, 2) == 12)
    assert (np.count_nonzero (idx_hs)), 'high-spin ground state not found'
    idx_hs = np.where (idx_hs)[0][0]
    e_hs = e_roots[idx_hs]
    idx_ls = (np.around (s2, 2) == 0)
    assert (np.count_nonzero (idx_ls)), 'low-spin ground state not found'
    idx_ls = np.where (idx_ls)[0][0]
    e_ls = e_roots[idx_ls]
    j = (e_ls - e_hs) / 12
    return j*au2cm

basis={'C': 'sto-3g','H': 'sto-3g','O': 'sto-3g','N': 'sto-3g','Cr': 'cc-pvdz'}
mol=gto.M (atom='cr2o3n6h21.xyz', verbose=4, spin=6, charge=3, basis=basis,
           output='cr2o3n6h21_lassis66_smallbasis.log')
mf=scf.ROHF(mol)
mf.chkfile = 'cr2o3n6h21_lassis66_smallbasis.chk'
mf.init_guess = 'chk'
mf.kernel()
assert (mf.converged)

# Make sure the overall 2*Sz (total neleca - total nelecb) is as small as possible (0 or 1)
las = LASSCF(mf,(6,6),((3,0),(0,3)),spin_sub=(4,4))

# Find the initial guess orbitals
try: # We only want the orbitals, not any of the other information on the chkfile
    mo_coeff = chkfile.load (las.chkfile, 'las')['mo_coeff']
except (OSError, TypeError, KeyError) as e: # First time through you have to make them from scratch
    from pyscf.mcscf import avas
    ncas_avas, nelecas_avas, mo_coeff = avas.kernel (mf, ['Cr 3d', 'Cr 4d'], minao=mol.basis)
    mc_avas = mcscf.CASCI (mf, ncas_avas, nelecas_avas)
    mo_list = mc_avas.ncore + np.array ([5,6,7,8,9,10,15,16,17,18,19,20])
    mo_coeff = las.sort_mo (mo_list, mo_coeff)
    mo_coeff = las.localize_init_guess (([0],[1]), mo_coeff)

# Direct exchange only result
las = lassi.states.spin_shuffle (las) # generate direct-exchange states
las.weights = [1.0/las.nroots,]*las.nroots # set equal weights
las.kernel (mo_coeff) # optimize orbitals
lsi0 = lassi.LASSI (las).run ()
print (("Direct exchange only is modeled by {} states constructed with "
        "lassi.states.spin_shuffle").format (las.nroots))
print ("J(LASSI, direct) = %.2f cm^-1" % yamaguchi (lsi0.e_roots, lsi0.s2))
print ("{} rootspaces and {} product states total\n".format (lsi0.nroots, lsi0.si.shape[1]))

# CASCI result for reference
mc = mcscf.CASCI (mf, 12, (6,0))
mc.kernel (las.mo_coeff)
e_roots = [mc.e_tot,]
s2 = [12,]
mc = mcscf.CASCI (mf, 12, (3,3))
mc.fix_spin_(ss=0)
mc.kernel (las.mo_coeff)
e_roots += [mc.e_tot]
s2 += [0,]
print ("J(CASCI) = %.2f cm^-1" % yamaguchi (np.asarray (e_roots), np.asarray (s2)))
print ("{} spin-alpha determinants * {} spin-beta determinants = {} determinants total\n".format (
    mc.ci.shape[0], mc.ci.shape[1], mc.ci.size))

# Direct exchange & kinetic exchange result
las = lassi.states.all_single_excitations (las) # generate kinetic-exchange states
print (("Use of lassi.states.all_single_excitations generates "
        "{} additional kinetic-exchange (i.e., charge-transfer) "
        "states").format (las.nroots-4))
las.lasci () # do not reoptimize orbitals at this step - not likely to converge
lsi1 = lassi.LASSI (las).run ()
print ("J(LASSI, direct & kinetic) = %.2f cm^-1" % yamaguchi (lsi1.e_roots, lsi1.s2))
print ("{} rootspaces and {} product states total\n".format (lsi1.nroots, lsi1.si.shape[1]))

# Locally excited states
print (("Including up to second locally-excited states improves "
        "results still further"))
lroots = np.minimum (3, las.get_ugg ().ncsf_sub)
las.lasci (lroots=lroots)
lsi2 = lassi.LASSI (las).run ()
print ("J(LASSI, direct & kinetic, nmax=2) = %.2f cm^-1" % yamaguchi (lsi2.e_roots, lsi2.s2))
print ("{} rootspaces and {} product states total".format (lsi2.nroots, lsi2.si.shape[1]))
print (("The first occurrence of the line 'Analyzing LASSI vectors for states = [0, 1, 2, 3]' "
        "in file {} begins the output of lassi.sitools.analyze for this calculation\n").format (
        mol.output))
lsi2.analyze (state=[0,1,2,3]) # Four states involved in this Yamaguchi manifold

# LASSIS
print ("LASSIS builds the entire model space automatically for you.")
print ("Because of this, you must initialize it with a LAS reference that has only 1 state in it")
las1 = LASSCF(mf,(6,6),((3,0),(0,3)),spin_sub=(4,4))
las1.lasci_(las.mo_coeff) # trailing underscore sets las1.mo_coeff = las.mo_coeff
lsi3 = lassi.LASSIS (las1).run (opt=0)
print ("J(LASSIS) = %.2f cm^-1" % yamaguchi (lsi3.e_roots, lsi3.s2))
print ("{} rootspaces and {} product states total".format (lsi3.nroots, lsi3.si.shape[1]))
print (("The second occurrence of the line 'Analyzing LASSI vectors for states = [0, 1, 2, 3]' "
        "in file {} begins the output of lassi.sitools.analyze for this calculation").format (
        mol.output))
lsi3.analyze (state=[0,1,2,3]) # Four states involved in this Yamaguchi manifold



