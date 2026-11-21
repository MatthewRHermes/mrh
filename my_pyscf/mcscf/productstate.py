# Moved to PySCF-Forge in the LASSCF migration (see MIGRATION.md).
# This module re-exports all of pyscf.mcscf.las.productstate (including private names) so that existing
# imports keep working.
import pyscf.mcscf.las.productstate as _forge
globals ().update ({k: v for k, v in vars (_forge).items () if not k.startswith ('__')})
