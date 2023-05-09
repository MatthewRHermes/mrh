# Ohh boy
from mrh.my_pyscf.mcscf.mc1step_csf import fix_ci_response_csf
import copy

class _DFLASCI: # Tag
    pass

def constrCASSCF(mf, ncas, nelecas, **kwargs):
    from pyscf import scf
    from pyscf.mcscf import addons
    from mrh.my_pyscf.mcscf import mc1step_constrained
    mf = scf.addons.convert_to_rhf(mf)
    return mc1step_constrained.CASSCF (mf, ncas, nelecas, **kwargs)

constrRCASSCF = constrCASSCF


# LAS-PDFT class
def LASPDFT(mc_class, ot, ncas, nelecas, ncore=None, frozen=None, verbose=5,
             **kwargs):
    
    try: import mrh
    except ImportError:
        print("For performing LASPDFT, you will require mrh")
        print('mrh can be found at: https://github.com/MatthewRHermes/mrh')
        raise
    try: from pyscf.mcpdft.mcpdft import get_mcpdft_child_class
    except ImportError:
         print("For performing LASPDFT, you will require pyscf-forge")
         print('pyscf-forge can be found at : https://github.com/pyscf/pyscf-forge')
         raise

    if isinstance(mc_class, (mrh.my_pyscf.mcscf.lasscf_sync_o0.LASSCFNoSymm)):
      import mrh 
      from mrh.my_pyscf.mcscf.laspdft import LASPDFT
      # This will change the certain functions of mc_class
      lspdft = LASPDFT(mc_class)
    else:
      print("LAS-object is not provided")
      raise
    
    mc2 = get_mcpdft_child_class(mc_class, ot, **kwargs)
    mc2.verbose = verbose
    mc2.mo_coeff = mc_class.mo_coeff.copy()    
    mc2.ci = copy.deepcopy(mc_class.ci)
    mc2.converged = mc_class.converged
    mc2 = lspdft.update_kernel(mc2)

# To make it consistent   
# LASSCF = _LASPDFT
# LASCI = _LASPDFT