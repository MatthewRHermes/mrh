from rhf import monkeypatch_setup
monkeypatch_teardown = monkeypatch_setup ()
from pyscf import scf, gto, lib

mol = gto.Mole()
mol.atom = [
    ['O' , (0. , 0.     , 0.)],
    [1   , (0. , -0.757 , 0.587)],
    [1   , (0. , 0.757  , 0.587)] ]
mol.basis = '631g'
mol.build()
mf = scf.RHF(mol).density_fit(auxbasis='ccpvdz-jkfit').run()
g = mf.nuc_grad_method ().set(auxbasis_response=not False).kernel()
print(lib.finger(g) - 0.0055166381900824879)
monkeypatch_teardown ()
g = mf.nuc_grad_method ().kernel()
print(lib.finger(g) - 0.005516638190173352)
print(abs(g-scf.RHF(mol).run().nuc_grad_method ().kernel()).max())
# -0.0000000000    -0.0000000000    -0.0241140368
#  0.0000000000     0.0043935801     0.0120570184
#  0.0000000000    -0.0043935801     0.0120570184

mfs = mf.as_scanner()
e1 = mfs([['O' , (0. , 0.     , 0.001)],
          [1   , (0. , -0.757 , 0.587)],
          [1   , (0. , 0.757  , 0.587)] ])
e2 = mfs([['O' , (0. , 0.     ,-0.001)],
          [1   , (0. , -0.757 , 0.587)],
          [1   , (0. , 0.757  , 0.587)] ])
print((e1-e2)/0.002*lib.param.BOHR)

