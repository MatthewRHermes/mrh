# Lahh dee dah
import copy
import numpy as np
from pyscf.mcpdft.mcpdft import get_mcpdft_child_class
from pyscf import mcscf, gto
from pyscf.lib import logger
from pyscf.mcscf import mc1step, casci
import mrh
from mrh.util.io import mcpdft_removal_warn
from mrh.my_pyscf.mcscf.lasscf_sync_o0 import LASSCFNoSymm, LASSCFSymm
from mrh.my_pyscf.mcscf.lasscf_async import LASSCFNoSymm as asyncLASSCFNoSymm, LASSCFSymm as asyncLASSCFSymm
mcpdft_removal_warn()


# NOTE: As of 02/06/2022, initializing PySCF mcscf classes with a symmetry-enabled molecule
# doesn't work.

def _MCPDFT(mc_class, mc_or_mf_or_mol, ot, ncas, nelecas, ncore=None, frozen=None,
            **kwargs):

    if isinstance(mc_or_mf_or_mol, (mc1step.CASSCF, casci.CASCI)):
        mc0 = mc_or_mf_or_mol
        mf_or_mol = mc_or_mf_or_mol._scf
    else:
        mc0 = None
        mf_or_mol = mc_or_mf_or_mol

    # Redefine the otfnal class for periodic systems
    ot = periodicpdft(mc_or_mf_or_mol, ot)

    if isinstance(mf_or_mol, gto.Mole) and mf_or_mol.symmetry:
        logger.warn(mf_or_mol,
                    'Initializing MC-SCF with a symmetry-adapted Mole object may not work!')
    if frozen is not None:
        mc1 = mc_class(mf_or_mol, ncas, nelecas, ncore=ncore, frozen=frozen)
    else:
        mc1 = mc_class(mf_or_mol, ncas, nelecas, ncore=ncore)

    mc2 = get_mcpdft_child_class(mc1, ot, **kwargs)

    if mc0 is not None:
        mc2.mo_coeff = mc_or_mf_or_mol.mo_coeff.copy()
        mc2.ci = copy.deepcopy(mc_or_mf_or_mol.ci)
        mc2.converged = mc0.converged
    return mc2


def CASSCFPDFT(mc_or_mf_or_mol, ot, ncas, nelecas, ncore=None, frozen=None,
               **kwargs):
    return _MCPDFT(mcscf.CASSCF, mc_or_mf_or_mol, ot, ncas, nelecas, ncore=ncore, frozen=frozen,
                   **kwargs)


def CASCIPDFT(mc_or_mf_or_mol, ot, ncas, nelecas, ncore=None, **kwargs):
    return _MCPDFT(mcscf.CASCI, mc_or_mf_or_mol, ot, ncas, nelecas, ncore=ncore,
                   **kwargs)


CASSCF = CASSCFPDFT
CASCI = CASCIPDFT


# LAS-PDFT
def _laspdftEnergy(mc_class, mc_or_mf_or_mol, ot, ncas_sub, nelecas_sub, DoLASSI=False, ncore=None, spin_sub=None,
                   frozen=None, **kwargs):
    if isinstance(mc_or_mf_or_mol, (LASSCFNoSymm, LASSCFSymm)) or \
        isinstance(mc_or_mf_or_mol, (asyncLASSCFNoSymm, asyncLASSCFSymm)):
        mc0 = mc_or_mf_or_mol
        mf_or_mol = mc_or_mf_or_mol._scf
    else:
        mc0 = None
        mf_or_mol = mc_or_mf_or_mol

    # Redefine the otfnal class for periodic systems
    ot = periodicpdft(mc_or_mf_or_mol, ot)

    if isinstance(mf_or_mol, gto.Mole) and mf_or_mol.symmetry:
        logger.warn(mf_or_mol,
                    'Initializing MC-SCF with a symmetry-adapted Mole object may not work!')

    if isinstance(mc_or_mf_or_mol, (LASSCFNoSymm, LASSCFSymm)) or \
        isinstance(mc_or_mf_or_mol, (asyncLASSCFNoSymm, asyncLASSCFSymm)):
        mc1 = mc_or_mf_or_mol
        if frozen is not None:
            mc1.frozen = frozen
    else:
        if frozen is not None:
            mc1 = mc_class(mf_or_mol, ncas_sub, nelecas_sub, ncore=ncore, spin_sub=spin_sub, frozen=frozen)
        else:
            mc1 = mc_class(mf_or_mol, ncas_sub, nelecas_sub, ncore=ncore, spin_sub=spin_sub)

    from mrh.my_pyscf.mcpdft.laspdft import get_mcpdft_child_class
    mc2 = get_mcpdft_child_class(mc1, ot, DoLASSI=DoLASSI, **kwargs)

    if mc0 is not None:
        mc2.mo_coeff = mc_or_mf_or_mol.mo_coeff.copy()
        mc2.ci = copy.deepcopy(mc_or_mf_or_mol.ci)
        mc2.converged = mc0.converged
    return mc2


def _lassipdftEnergy(mc_class, mc_or_mf_or_mol, ot, ncas_sub, nelecas_sub, DoLASSI=False, ncore=None, spin_sub=None,
                     frozen=None, states=None, **kwargs):
    from mrh.my_pyscf.lassi import lassi

    if isinstance(mc_or_mf_or_mol, lassi.LASSI):
        mc0 = mc_or_mf_or_mol._las
        mf_or_mol = mc_or_mf_or_mol._las._scf
    else:
        raise "Requires lassi instance"

    mc1 = mc_class(mf_or_mol, ncas_sub, nelecas_sub, ncore=ncore, spin_sub=spin_sub)

    # Redefine the otfnal class for periodic systems
    ot = periodicpdft(mc_or_mf_or_mol, ot)

    from mrh.my_pyscf.mcpdft.laspdft import get_mcpdft_child_class
    mc2 = get_mcpdft_child_class(mc1, ot, DoLASSI=DoLASSI, states=states, **kwargs)

    if mc0 is not None:
        mc2.mo_coeff = mc_or_mf_or_mol.mo_coeff.copy()
        mc2.ci = copy.deepcopy(mc_or_mf_or_mol.ci)
        mc2.converged = mc0.converged
        _keys = mc_or_mf_or_mol._keys.copy()
        mc2.__dict__.update(mc_or_mf_or_mol.__dict__)
        mc2._keys = mc2._keys.union(_keys)
        mc2.e_mcscf = np.average(mc_or_mf_or_mol.e_roots).copy()
    return mc2


def LASSCFPDFT(mc_or_mf_or_mol, ot, ncas_sub=None, nelecas_sub=None, ncore=None, spin_sub=None, frozen=None,
               **kwargs):
    if ncas_sub is None: ncas_sub = getattr(mc_or_mf_or_mol, 'ncas_sub', None)
    if nelecas_sub is None: nelecas_sub = getattr(mc_or_mf_or_mol, 'nelecas_sub', None)
    from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
    return _laspdftEnergy(LASSCF, mc_or_mf_or_mol, ot, ncas_sub, nelecas_sub, ncore=ncore,
                          spin_sub=spin_sub, frozen=frozen, **kwargs)


def LASSIPDFT(mc_or_mf_or_mol, ot, ncas_sub=None, nelecas_sub=None, ncore=None, spin_sub=None,
              frozen=None, states=None, **kwargs):
    if ncas_sub is None: ncas_sub = getattr(mc_or_mf_or_mol, 'ncas_sub', None)
    if nelecas_sub is None: nelecas_sub = getattr(mc_or_mf_or_mol, 'nelecas_sub', None)
    from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
    return _lassipdftEnergy(LASSCF, mc_or_mf_or_mol, ot, ncas_sub, nelecas_sub, DoLASSI=True, ncore=ncore,
                            spin_sub=spin_sub, frozen=frozen, states=states, **kwargs)


LASSCF = LASSCFPDFT
LASSI = LASSIPDFT
LASSIS = LASSIPDFT


def CIMCPDFT(*args, **kwargs):
    from mrh.my_pyscf.mcpdft.var_mcpdft import CIMCPDFT as fn
    return fn(mcscf.CASCI, *args, **kwargs)


def CIMCPDFT_SCF(*args, **kwargs):
    from mrh.my_pyscf.mcpdft.var_mcpdft import CIMCPDFT as fn
    return fn(mcscf.CASSCF, *args, **kwargs)

def _getmole(mc_or_mf_mol):
    '''
    A function to get the mol object from the mc_or_mf_mol object
    '''
    from pyscf.pbc import gto as pbcgto
    if isinstance(mc_or_mf_mol, (mc1step.CASSCF, casci.CASCI)):
        return mc_or_mf_mol._scf.mol
    elif isinstance(mc_or_mf_mol, gto.Mole) or isinstance(mc_or_mf_mol, pbcgto.cell.Cell):
        return mc_or_mf_mol
    else:
        return mc_or_mf_mol.mol
    

def periodicpdft(mc_or_mf_mol, ot):
    """
    A wrapper function to check whether the mol is cell object or not.
    if yes, then the otfnal class will be modified.
    
    Note: Remember to write the sanity check for the kpts.
    
    Also, this is not the best way to handle the situation but it works for now.In future,
    I will put this in mrh.pbc folder.

    Args:
        mc: mcscf object
    """
    mol = _getmole(mc_or_mf_mol)
    assert isinstance(ot, str), "The ot should be a string"
    from pyscf.pbc import gto
    if isinstance(mol, gto.cell.Cell):
        from mrh.my_pyscf.mcpdft.otfnalperiodic import _get_transfnal
        sanity_check_for_kpts(mc_or_mf_mol)
        fnal = _get_transfnal(mc_or_mf_mol, ot)
        return fnal
    else:
        return ot

def sanity_check_for_kpts(mc_or_mf_mol):
    '''
    A function to check total number of kpts if it is greater than 1
    raise an error.
    '''
    if hasattr(mc_or_mf_mol, '_scf'):
        nkpts = len(mc_or_mf_mol._scf.kpts)
    
    elif hasattr(mc_or_mf_mol, '_las'):
        nkpts = len(mc_or_mf_mol._las._scf.kpts)

    elif hasattr(mc_or_mf_mol, 'kpts'):
        nkpts = len(mc_or_mf_mol.kpts)

    else:
        raise NotImplementedError ("The input object does not have kpts attribute")
    
    if nkpts > 1:
        raise ValueError ("Only supercell calculations can be performend with MCPDFT")
    
    
    
