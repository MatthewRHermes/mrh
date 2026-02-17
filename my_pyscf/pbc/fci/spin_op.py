
import numpy as np
from pyscf.fci.spin_op import contract_ss

# Spin square operator for the complex FCI wave function.
def spin_square0(fcivec, norb, nelec):
    '''
    Spin square for complex RHF-FCI CI wfn only.
    (a-ib)*S^2*(a+ib) = a*S^2*a + b*S^2*b
    '''
    assert fcivec.dtype == np.complex128
    def s2(ci1):
        ci1ssket = contract_ss(ci1, norb, nelec)
        return np.vdot(ci1, ci1ssket)
    
    ss = s2(fcivec.real)
    ss += s2(fcivec.imag)
    
    s = np.sqrt(ss + 0.25) - 0.5
    multip = 2*s + 1
    return ss, multip