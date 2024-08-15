# Lahh dee dah
import copy
import numpy as np
from pyscf.mcpdft.mcpdft import get_mcpdft_child_class
from pyscf import mcscf, gto, lib
from pyscf.lib import logger
from pyscf.mcscf import mc1step, casci
import h5py
from mrh.util.io import mcpdft_removal_warn
from mrh.my_pyscf.mcscf.lasscf_sync_o0 import LASSCFNoSymm, LASSCFSymm

import tempfile

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
    if isinstance(mc_or_mf_or_mol, (LASSCFNoSymm, LASSCFSymm)):
        mc0 = mc_or_mf_or_mol
        mf_or_mol = mc_or_mf_or_mol._scf
    else:
        mc0 = None
        mf_or_mol = mc_or_mf_or_mol
    if isinstance(mf_or_mol, gto.Mole) and mf_or_mol.symmetry:
        logger.warn(mf_or_mol,
                    'Initializing MC-SCF with a symmetry-adapted Mole object may not work!')

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


def _prelassipdftenergy(mc_class, mc_or_mf_or_mol, ot, ncas_sub, nelecas_sub, DoPreLASSI=False, ncore=None,
                        spin_sub=None,
                        frozen=None, **kwargs):
    if isinstance(mc_or_mf_or_mol, (LASSCFNoSymm, LASSCFSymm)):
        mf_or_mol = mc_or_mf_or_mol._scf
    else:
        raise NotImplementedError

    rdmstmpfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
    with h5py.File(rdmstmpfile, 'w') as f:
        rdm1s = mc_or_mf_or_mol.state_make_casdm1s()
        for i in range(0, len(mc_or_mf_or_mol.ci[0])):
            rdm1s_dname = f'rdm1s_{i}'
            f.create_dataset(rdm1s_dname, data=rdm1s)
            rdm2 = mc_or_mf_or_mol.state_make_casdm2(state=i)
            rdm2s_dname = f'rdm2s_{i}'
            f.create_dataset(rdm2s_dname, data=rdm2)

    if frozen is not None:
        mc1 = mc_class(mf_or_mol, ncas_sub, nelecas_sub, ncore=ncore, spin_sub=spin_sub, frozen=frozen)
    else:
        mc1 = mc_class(mf_or_mol, ncas_sub, nelecas_sub, ncore=ncore, spin_sub=spin_sub)

    from mrh.my_pyscf.mcpdft.laspdft import get_mcpdft_child_class
    mc2 = get_mcpdft_child_class(mc1, ot, DoPreLASSI=DoPreLASSI, rdmstmpfile=rdmstmpfile, **kwargs)

    # Don't have to make the copies of CI vectors. Unnecessarily will
    # increase the memory footprint.
    # if mc0 is not None:
    #     mc2.mo_coeff = mc_or_mf_or_mol.mo_coeff.copy()
    #     mc2.ci = copy.deepcopy(mc_or_mf_or_mol.ci)
    #     mc2.converged = mc0.converged
    return mc2


def LASSCFPDFT(mc_or_mf_or_mol, ot, ncas_sub=None, nelecas_sub=None, ncore=None, DoPreLASSI=False,
               spin_sub=None, frozen=None, **kwargs):
    from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
    if ncas_sub is None: ncas_sub = getattr(mc_or_mf_or_mol, 'ncas_sub', None)
    if nelecas_sub is None: nelecas_sub = getattr(mc_or_mf_or_mol, 'nelecas_sub', None)
    if DoPreLASSI:
        
        return _prelassipdftenergy(mc_or_mf_or_mol, ot, ncas_sub, nelecas_sub, DoPreLASSI=True, ncore=ncore,
                                   spin_sub=spin_sub, frozen=frozen, **kwargs)
    else:
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
LASSIS = LASSIPDFT  # Not Sure acc. to issue #34 of mrh


def CIMCPDFT(*args, **kwargs):
    from mrh.my_pyscf.mcpdft.var_mcpdft import CIMCPDFT as fn
    return fn(mcscf.CASCI, *args, **kwargs)


def CIMCPDFT_SCF(*args, **kwargs):
    from mrh.my_pyscf.mcpdft.var_mcpdft import CIMCPDFT as fn
    return fn(mcscf.CASSCF, *args, **kwargs)
