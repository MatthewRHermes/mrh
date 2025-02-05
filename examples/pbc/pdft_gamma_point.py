
'''
Here, I am doing the gamma-point CASSCF followed by PDFT.
Note: 
    1. mf.exxdiv=None should be used. Post-SCF method require this.
    2. I am using GDF. In case of default DF which is FFTDF, the grids 
    for the periodic DFT will change. Which I haven't tested yet.
    3. Also, the MCPDFT (tPBE) == PBE with SD active space and same mo_coeff.
'''


POSCAR='''
Polyacetylene Unit Cell
1.0
2.4700000000  0.0000000000  0.0000000000
0.0000000000  17.5000000000  0.0000000000
0.0000000000  0.0000000000  17.5000000000
C    H
2   2
Cartesian
-0.5892731038  0.3262391909  0.0000000000
0.5916281105  -0.3261693897  0.0000000000
-0.5866101958  1.4126530287  0.0000000000
0.5889652025  -1.4125832275  0.0000000000
'''

import numpy as np
from pyscf.pbc import gto, scf, dft
from pyscf import mcscf
from mrh.my_pyscf import mcpdft

# Periodic Calculation for CH=CH uni, using CASCI vs RHF
def getcell():
    cell = gto.Cell()
    cell.a='''
    2.4700000000  0.0000000000  0.0000000000
    0.0000000000  17.5000000000  0.0000000000
    0.0000000000  0.0000000000  17.5000000000
    '''
    cell.atom='''
    C -0.5892731038  0.3262391909  0.0000000000
    C 0.5916281105  -0.3261693897  0.0000000000
    H -0.5866101958  1.4126530287  0.0000000000
    H 0.5889652025  -1.4125832275  0.0000000000
    '''
    cell.basis = '321g'
    cell.precision=1e-12
    cell.verbose = 4
    cell.build()
    return cell

def periodicDFT():
    cell = getcell()
    mf = scf.RKS(cell).density_fit()
    mf.verbose=4
    mf.exxdiv=None
    mf.xc='pbe'
    eperpbe = mf.kernel()
    mc = mcpdft.CASCI(mf, 'tPBE', 1,2)
    epdftper = mc.kernel(mf.mo_coeff)[0]
    print("Periodic Cal (PBE vs tPBE): ", np.allclose(eperpbe,epdftper, 1e-7))

def periodicHF():
    cell = getcell()
    mf = scf.RHF(cell).density_fit()
    mf.exxdiv=None
    eper = mf.kernel()
    mc = mcscf.CASCI(mf, 1,2)
    ecasper = mc.kernel(mf.mo_coeff)[0]
    print("Periodic Cal (CASCI vs RHF): ", np.allclose(eper,ecasper, 1e-7))

if __name__ == "__main__":
    periodicHF()
    periodicDFT() 
