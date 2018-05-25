# Ohh boy
from mrh.my_pyscf.mcscf import mc1step_constrained
from pyscf.mcscf import addons

def constrCASSCF(mf, ncas, nelecas, **kwargs):
    from pyscf import scf
    mf = scf.addons.convert_to_rhf(mf)
    return mc1step_constrained.CASSCF (mf, ncas, nelecas, **kwargs)

constrRCASSCF = constrCASSCF


