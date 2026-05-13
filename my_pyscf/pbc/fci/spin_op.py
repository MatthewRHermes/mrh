
import sys
import numpy as np

from pyscf import lib
from pyscf.fci.spin_op import contract_ss

# Author: Bhavnesh Jangid

# Spin operators for complex FCI vectors.

def contract_ss0(fcivec, norb, nelec, **kwargs):
    '''
    To implement the spin_penalty method, I will need this function.
    Basically, it applies the S^2 operator to a complex CI vector.
    For fcivec = a + i b,
        S^2(fcivec) = S^2(a) + i S^2(b)
    '''
    assert fcivec.dtype == np.complex128
    ci1_real = contract_ss(fcivec.real, norb, nelec)
    ci1_imag = contract_ss(fcivec.imag, norb, nelec)
    ci1 = ci1_real.astype(fcivec.dtype)
    ci1.real = ci1_real
    ci1.imag = ci1_imag
    ci1_real = ci1_imag = None
    return ci1

def spin_square0(fcivec, norb, nelec, **kwargs):
    '''
    Spin square for complex RHF-FCI CI wfn only.
    (a-ib)*S^2*(a+ib) = a*S^2*a + b*S^2*b + i(a*S^2*b - b*S^2*a)
    '''
    assert fcivec.dtype == np.complex128
    verbose = kwargs.get('verbose', 0)
    log = lib.logger.Logger(sys.stdout, verbose)

    def s2(ci1, ci2):
        ci1ssket = contract_ss(ci1, norb, nelec)
        return np.vdot(ci2, ci1ssket)
    
    ssreal = s2(fcivec.real, fcivec.real)
    ssreal += s2(fcivec.imag, fcivec.imag)
    
    ssimag = (s2(fcivec.real, fcivec.imag) 
              - s2(fcivec.imag, fcivec.real))

    if abs(ssimag) > 1e-3:
        log.warn("Spin square is not real. Imaginary part = %s", ssimag)

    ss = ssreal
    s = np.sqrt(ss + 0.25) - 0.5
    multip = 2*s + 1

    # Although, I can take the sqrt for the complex numbers as well.
    if verbose >= lib.logger.DEBUG:
        sstot = ssreal + 1j*ssimag
        stot = np.sqrt(sstot + 0.25) - 0.5
        multip_tot = 2*stot + 1
        log.debug("Spin expectation value including the complex part")
        log.debug("Spin square = %s, Spin = %s, Multiplicity = %s", sstot, stot, multip_tot)

    return ss, multip
