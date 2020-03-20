# Ohh boy
from mrh.my_pyscf.mcscf import mc1step_constrained
from mrh.my_pyscf.mcscf.mc1step_csf import fix_ci_response_csf
from mrh.my_pyscf.grad import sacasscf as sacasscf_grad
from pyscf import mcscf as pyscf_mcscf

def constrCASSCF(mf, ncas, nelecas, **kwargs):
    from pyscf import scf
    mf = scf.addons.convert_to_rhf(mf)
    return mc1step_constrained.CASSCF (mf, ncas, nelecas, **kwargs)

constrRCASSCF = constrCASSCF

def CASSCF(mf_or_mol, ncas, nelecas, ncore=None, frozen=None):
    mc = pyscf_mcscf.CASSCF (mf_or_mol, ncas, nelecas, ncore=None, frozen=None)
    class dressedCASSCF (mc.__class__):
        def __init__(self, my_mc):
            self.__dict__.update (my_mc.__dict__)
        def nuc_grad_method (self, state=None):
            if isinstance (self, pyscf_mcscf.StateAverageMCSCFSolver):
                from mrh.my_pyscf.grad import sacasscf as sacasscf_grad
                return sacasscf_grad.Gradients (self, state=state)
            else:
                return super().nuc_grad_method ()
    return dressedCASSCF (mc)

