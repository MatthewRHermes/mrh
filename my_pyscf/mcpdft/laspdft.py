# Moved to PySCF-Forge in the LASSCF migration (see MIGRATION.md). LASSI-PDFT stays in mrh (lassipdft.py).
# This module re-exports all of pyscf.mcpdft.laspdft (including private names) so that existing
# imports keep working.
import pyscf.mcpdft.laspdft as _forge
globals ().update ({k: v for k, v in vars (_forge).items () if not k.startswith ('__')})

def get_mcpdft_child_class(mc, ot, DoLASSI=False, states=None,
                           _pdft_class=_LASPDFT, **kwargs):
    if DoLASSI:
        from mrh.my_pyscf.mcpdft.lassipdft import get_mcpdft_child_class as get_lassipdft_child_class
        return get_lassipdft_child_class(mc, ot, states=states, _pdft_class=_pdft_class, **kwargs)

    PDFT = _get_mcpdft_child_class(mc, _pdft_class=_pdft_class)
    if mc.ci is not None:
        mc.fcisolver.nroots = mc.nroots
    else:
        mc.fcisolver.nroots = 1
    return _instantiate(PDFT, mc, ot, **kwargs)
