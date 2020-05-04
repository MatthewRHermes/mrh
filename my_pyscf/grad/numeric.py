import numpy as np
from pyscf.grad import rhf as rhf_grad
from pyscf.lib import param, logger

STEPSIZE_DEFAULT=0.001
SCANNER_VERBOSE_DEFAULT=0

# MRH 05/04/2020: I don't know why I have to present the molecule instead
# of just the coordinates, but somehow I can't get the units right any other
# way.

def _make_mol (mol, coords):
    return [[mol.atom_symbol (i), coords[i,:]] for i in range (mol.natm)]

def _numgrad_1df (mol, scanner, coords, iatm, icoord, delta=0.001):
    coords[iatm,icoord] += delta
    ep = scanner (_make_mol (mol, coords))
    coords[iatm,icoord] -= 2*delta
    em = scanner (_make_mol (mol, coords))
    coords[iatm,icoord] += delta
    return (ep-em) / (2*delta)*param.BOHR


class Gradients (rhf_grad.GradientsBasics):

    def __init__(self, method, stepsize=STEPSIZE_DEFAULT, scanner_verbose=SCANNER_VERBOSE_DEFAULT):
        self.stepsize = stepsize
        self.scanner = method.as_scanner ()
        self.scanner.verbose = scanner_verbose
        # MRH 05/04/2020: there must be a better way to do this
        if hasattr (self.scanner, '_scf'):
            self.scanner._scf.verbose = scanner_verbose
        rhf_grad.GradientsBasics.__init__(self, method)

    def _numgrad_1df (self, iatm, icoord):
        return _numgrad_1df (self.mol, self.scanner, self.mol.atom_coords () * param.BOHR,
            iatm, icoord, delta=self.stepsize)

    def kernel (self, atmlst=None, stepsize=None):
        if atmlst is None:
            atmlst = self.atmlst
        if stepsize is None:
            stepsize = self.stepsize
        else:
            self.stepsize = stepsize
        if atmlst is None:
            atmlst = list (range (self.mol.natm))
        
        coords = self.mol.atom_coords () * param.BOHR
        de = [[self._numgrad_1df (i, j) for j in range (3)] for i in atmlst]
        self.de = np.asarray (de)
        self._finalize ()
        return self.de

    def grad_elec (self, atmlst=None, stepsize=None):
        # This is just computed backwards from full gradients
        if atmlst is None:
            atmlst = self.atmlst
        if stepsize is None:
            stepsize = self.stepsize
        else:
            self.stepsize = stepsize
        if atmlst is None:
            atmlst = list (range (self.mol.natm))

        if getattr (self, 'de', None) is not None:
            de = self.de = self.kernel (atmlst=atmlst, stepsize=stepsize)
        de_elec = de - self.grad_nuc (atmlst=atmlst)
        return de_elec

    def _finalize(self):
        if self.verbose >= logger.NOTE:
            logger.note(self, '----------- %s numeric gradients -------------',
                        self.base.__class__.__name__)
            self._write(self.mol, self.de, self.atmlst)
            logger.note(self, '----------------------------------------------')


