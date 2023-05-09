# Lahh dee dah
import copy
from pyscf.mcpdft.mcpdft import get_mcpdft_child_class
from pyscf.mcpdft.otfnal import make_hybrid_fnal as hyb
from pyscf import mcscf, gto
from pyscf.lib import logger
from pyscf.mcscf import mc1step, casci
from types import MethodType
import mrh
from mrh.util.io import mcpdft_removal_warn
mcpdft_removal_warn ()

# NOTE: As of 02/06/2022, initializing PySCF mcscf classes with a symmetry-enabled molecule
# doesn't work.

def _MCPDFT (mc_class, mc_or_mf_or_mol, ot, ncas, nelecas, ncore=None, frozen=None,
             **kwargs):
    if isinstance (mc_or_mf_or_mol, (mc1step.CASSCF, casci.CASCI)):
        mc0 = mc_or_mf_or_mol
        mf_or_mol = mc_or_mf_or_mol._scf
    else:
        mc0 = None
        mf_or_mol = mc_or_mf_or_mol
    if isinstance (mf_or_mol, gto.Mole) and mf_or_mol.symmetry:
        logger.warn (mf_or_mol,
                     'Initializing MC-SCF with a symmetry-adapted Mole object may not work!')
    if frozen is not None: mc1 = mc_class (mf_or_mol, ncas, nelecas, ncore=ncore, frozen=frozen)
    else: mc1 = mc_class (mf_or_mol, ncas, nelecas, ncore=ncore)
    mc2 = get_mcpdft_child_class (mc1, ot, **kwargs)
    if mc0 is not None:
        mc2.mo_coeff = mc_or_mf_or_mol.mo_coeff.copy ()    
        mc2.ci = copy.deepcopy (mc_or_mf_or_mol.ci)
        mc2.converged = mc0.converged
    return mc2

def CASSCFPDFT (mc_or_mf_or_mol, ot, ncas, nelecas, ncore=None, frozen=None,
                **kwargs):
    return _MCPDFT (mcscf.CASSCF, mc_or_mf_or_mol, ot, ncas, nelecas, ncore=ncore, frozen=frozen,
                    **kwargs)

def CASCIPDFT (mc_or_mf_or_mol, ot, ncas, nelecas, ncore=None, **kwargs):
    return _MCPDFT (mcscf.CASCI, mc_or_mf_or_mol, ot, ncas, nelecas, ncore=ncore,
                    **kwargs)

CASSCF=CASSCFPDFT
CASCI=CASCIPDFT


# LAS-PDFT
def _laspdftEnergy(mc_class, ot, ncas, nelecas, ncore=None, frozen=None, verbose=5,
             **kwargs):
    try: 
        from pyscf.mcpdft.mcpdft import get_mcpdft_child_class
    except ImportError:
         msg = "For performing LASPDFT, you will require pyscf-forge.\n" +\
         "pyscf-forge can be found at : https://github.com/pyscf/pyscf-forge"
         raise ImportError(msg)

    if isinstance(mc_class, (mrh.my_pyscf.mcscf.lasscf_sync_o0.LASSCFNoSymm)):
        from mrh.my_pyscf.mcpdft.laspdft import laspdfthelper
        # This will change the certain functions of mc_class
        laspdfthelper(mc_class)
    else:
        raise ValueError("LAS-object is not provided")
    
    mc2 = get_mcpdft_child_class(mc_class, ot, **kwargs)
    mc2.verbose = verbose
    mc2.mo_coeff = mc_class.mo_coeff.copy()    
    mc2.ci = copy.deepcopy(mc_class.ci)
    mc2.converged = mc_class.converged
    mc2.kernel = MethodType(laspdft_kernel, mc2)
    return mc2

def laspdft_kernel(self, mo_coeff=None, ci0=None, otxc=None, grids_attr=None,
        grids_level=None, **kwargs ):
    #TODO
    #self.optimize_mcscf_(mo_coeff=mo_coeff, ci0=ci0, **kwargs)
    self.compute_pdft_energy_(otxc=otxc, grids_attr=grids_attr,
                                grids_level=grids_level, **kwargs)
    return (self.e_tot, self.e_ot, self.e_mcscf, self.e_cas, self.ci,
    self.mo_coeff, self.mo_energy)

LASSCF = _laspdftEnergy
LASCI = _laspdftEnergy

def CIMCPDFT (*args, **kwargs):
    from mrh.my_pyscf.mcpdft.var_mcpdft import CIMCPDFT as fn
    return fn (mcscf.CASCI, *args, **kwargs)
def CIMCPDFT_SCF (*args, **kwargs):
    from mrh.my_pyscf.mcpdft.var_mcpdft import CIMCPDFT as fn
    return fn (mcscf.CASSCF, *args, **kwargs)


