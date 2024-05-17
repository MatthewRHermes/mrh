import numpy
from pyscf import gto, scf
from pyscf import mcscf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.tools.molden import from_lasscf

'''
Here I am doing first gamma point LASCI calculation
For reference I did molecular calculation, periodic calculation
with large unit cell (supercell) and at gamma point should be 
equals to the molecular values
Outcome of these calculations
1. HF, CACI, and LASCI is converging to molecular value

Molecular calculations are done without density fitting, but
for periodic calculations FFTDF is by default on.
Here i am using GDF
Probably that's the reason of slow convergence!
'''

# Molecular Calculation
mol = gto.M(atom = '''
    H         -6.37665        2.20769        0.00000
    H         -5.81119        2.63374       -0.00000
    ''',
    basis = '631g',
    verbose = 4,max_memory=10000)

mf = scf.RHF(mol)
memf = mf.kernel()

mc = mcscf.CASCI(mf, 2, 2)
mecasci = mc.kernel()[0]

las = LASSCF(mf, (2,), (2,))
mo0 = las.localize_init_guess((list(range(2)),))
melasci = las.lasci(mo0)

del mol, mf, mc, las

# Periodic Calculation
from pyscf.pbc import gto, scf

def cellObject(x):
    cell = gto.M(a = numpy.eye(3)*x,
    atom = '''
    H         -6.37665        2.20769        0.00000
    H         -5.81119        2.63374       -0.00000
    ''',
    basis = '631g',
    verbose = 1, max_memory=10000)
    cell.build()

    mf = scf.RHF(cell).density_fit() # GDF
    mf.exxdiv = None
    emf = mf.kernel()
    
    mc = mcscf.CASCI(mf, 2, 2)
    ecasci = mc.kernel()[0]

    las = LASSCF(mf, (2,), (2,))
    mo0 = las.localize_init_guess((list(range(2)), ))
    elasci = las.lasci(mo0)

    del cell, mc, mf, las
    return x, emf, ecasci, elasci

print(" Energy Comparision with cubic unit cell size ")
print(f"{'LatticeVector(a)':<20} {'HF':<20} {'CASCI':<15} {'LASCI':<20}")
print(f"{'Reference':<18} {memf:<18.9f} {mecasci:<18.9f} {melasci[1]:<18.9f}")

for x in range(3,17, 2):
    x, emf, ecasci, elasci = cellObject(x)
    print(f"{x:<18.1f} {emf:<18.9f} {ecasci:<18.9f} {elasci[1]:<18.9f}")
