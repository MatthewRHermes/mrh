# Lahh dee dah
import copy
from pyscf.mcpdft.mcpdft import get_mcpdft_child_class
from pyscf.mcpdft.otfnal import make_hybrid_fnal as hyb
from pyscf import mcscf, gto
from pyscf.lib import logger
from pyscf.mcscf import mc1step, casci
import mrh
from mrh.util.io import mcpdft_removal_warn
from types import MethodType

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
# MRH commentary: note that ncas, nelecas, ncore, and frozen aren't used at all in this function.
# Therefore, they do nothing and shouldn't appear in this argument list at all.
#
# Broader design commentary:
# The idea behind the CASSCF-MC-PDFT and CASCI-MC-PDFT implementations was for the user to go, i.e.,
#   mc = mcpdft.CASSCF (mf, 'tPBE', [remaining args of mcscf.CASSCF])
# where mf is a Hartree-Fock instance, such that mc.kernel () runs ~both~ the CASSCF/CASCI step
# ~and~ the MC-PDFT energy calculation step. The way you've set it up, however, the user has to
# do it in two steps: first set up and run a LASSCF calculation, and THEN instantiate a second,
# different method instance for the MC-PDFT energy calculation. That's valid, and maybe in the
# grand scheme of things it is even smarter, but note that it is different. The argument that
# you've called "mc_class" is not an MC-SCF class, the way it is in _MCPDFT above.
    
    if isinstance(mc_class, (mrh.my_pyscf.mcscf.lasscf_sync_o0.LASSCFNoSymm)): pass
    else: raise ValueError("LAS-object is not provided")
    
    from mrh.my_pyscf.mcpdft.laspdft import get_mcpdft_child_class

    mc2 = get_mcpdft_child_class(mc_class, ot, **kwargs)
    mc2.verbose = verbose
    mc2.mo_coeff = mc_class.mo_coeff.copy()
    mc2.ci = copy.deepcopy(mc_class.ci)
    mc2.converged = mc_class.converged
    return mc2

LASSCF = _laspdftEnergy
LASCI = _laspdftEnergy

def CIMCPDFT (*args, **kwargs):
    from mrh.my_pyscf.mcpdft.var_mcpdft import CIMCPDFT as fn
    return fn (mcscf.CASCI, *args, **kwargs)
def CIMCPDFT_SCF (*args, **kwargs):
    from mrh.my_pyscf.mcpdft.var_mcpdft import CIMCPDFT as fn
    return fn (mcscf.CASSCF, *args, **kwargs)


