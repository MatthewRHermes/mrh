# !/usr/bin/env python

import os
import numpy as np     
from pyscf import lib

from pyscf.tools.fcidump import write_head, DEFAULT_FLOAT_FORMAT, TOL

logger = lib.logger


# Author: Bhavnesh Jangid

# Interface to use the DMRG as the FCI solver.

# Note: For the complex integrals in DMRG-CI, I can not directly use the pyscf+dmrgscf+block2.
# As it doesn't support complex numbers. However, The block2 is really 
# good in handling various types of Hamiltonians symmetries. I am writing an interface
# with the SU2 symmetry along with the complex numbers. In the limit of bond dimension 
# going to infinity this would be equivalent to the exact FCI with complex integrals.
# I have structured most of these functions similar to the ones in pyscf+dmrgscf+block2, 
# so that it would be easier to maintain and update the code in the future.
# Dependency: This code would only require the block2 with USECOMPLEX=ON.
