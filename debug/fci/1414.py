import numpy as np
from pyscf import lib, gto, ao2mo
from mrh.my_pyscf.fci import csf_solver
mol = gto.M (atom='Si 0 0 0', spin=14, charge=0, output='1414.log', verbose=lib.logger.DEBUG1)
fci = csf_solver (mol, smult=1)
norb = 14
nelec = (7,7)
h1 = np.random.rand (14,14)
h1 += h1.T
h2 = np.random.rand (14,14,14,14)
h2 = ao2mo.restore (8, h2, norb)
h2 = ao2mo.restore (1, h2, norb)
e, ci = fci.kernel (h1, h2, norb, nelec)
print (e)


