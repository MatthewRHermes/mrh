

# For the complex integrals in DMRG-CI, I need to make a subclass of
# the DMRGCI class that can handle the complex number. The block2 is really 
# good in handling the SU2 symmetry along with the complex numbers.

# The PySCF tools have FCIDUMP reader and writer but that only workds for the real-numbers.
# I need to write the complex version of the FCIDUMP reader and writer.


# DMRG-CI with complex spatial integrals for c-CASCI and c-CASSCF calculations.
import os
import numpy as np     
from pyscf import lib
from pyscf.tools.fcidump import write_head, DEFAULT_FLOAT_FORMAT, TOL

logger = lib.logger

try:
    from pyscf.dmrgscf import DMRGCI
    from pyscf.dmrgscf import dmrg_sym
    from pyscf.dmrgscf.dmrgci import make_schedule, executeBLOCK, readEnergy
    from pyscf.dmrgscf.dmrgci import block_version, check_call
except ImportError:
    raise ImportError("dmrgscf is not installed. Please install dmrgscf and block2 with USECOMPLEX=ON to use DMRGCIComplex.")


def write_hcore(fout, h, ncas, tol=TOL, float_format=DEFAULT_FLOAT_FORMAT):
    '''
    Write the one-electron integrals to the FCIDUMP file.
    args:
        fout: file object 
            to write the integrals
        h: 2D array of shape (ncas, ncas) (np.complex128)
            the one-electron integrals in the wannier orbital basis.
        ncas: int
            the number of active orbitals. (ncas * nkpts)
        tol: float
            the tolerance for writing the integrals. 
        float_format: str
            the format for writing the real and imaginary parts of the integrals.
    '''
    h = h.reshape(ncas, ncas)
    output_format = f"{float_format} {float_format}  %4d  %4d  0  0\n"
    for i in range(ncas):
        for j in range(i + 1):
            hij = h[i, j]
            if abs(hij.real) > tol: # or abs(hij.imag) > tol:
                if abs(hij.imag) > tol:
                    hijimag = hij.imag
                else:
                    hijimag = 0.0
                fout.write(output_format % (hij.real, hijimag, i + 1, j + 1))

def write_eri(fout, eri, ncas, tol=TOL, float_format=DEFAULT_FLOAT_FORMAT):
    eri = eri.reshape(ncas, ncas, ncas, ncas)
    output_format = f"{float_format} {float_format} %4d %4d %4d %4d\n"
    for i in range(ncas):
        for j in range(i + 1):
            for k in range(i + 1):
                lmax = (j + 1) if (k == i) else (k + 1)
                for l in range(lmax):
                    v = eri[i, j, k, l]
                    if abs(v.real) > tol: # or abs(v.imag) > tol:
                        if abs(v.imag) > tol:
                            vimag = v.imag
                        else:
                            vimag = 0.0
                        fout.write(output_format % (v.real, vimag, i + 1, j + 1, k + 1, l + 1))

def from_integrals(integralFile, h1e, h2e, ncas, nelec, nuc=0, ms=0, orbsym=None,
                   tol=TOL, float_format=DEFAULT_FLOAT_FORMAT):
    with open(integralFile, 'w') as fout:
        write_head(fout, ncas, nelec, ms, orbsym)
        write_eri(fout, h2e, ncas, tol=tol, float_format=float_format)
        write_hcore(fout, h1e, ncas, tol=tol, float_format=float_format)
        output_format = f"{float_format}{float_format}  0  0  0  0\n"
        fout.write(output_format % (nuc.real, nuc.imag))

def writeDMRGConfFile(DMRGCIobj, nelec, Restart,
                    maxIter=None, with_2pdm=True, extraline=[]):
    confFile = os.path.join(DMRGCIobj.runtimeDir, DMRGCIobj.configFile)

    f = open(confFile, 'w')

    if isinstance(nelec, (int, np.integer)):
        nelecb = (nelec-DMRGCIobj.spin) // 2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec

    f.write('nelec %i\n'%(neleca+nelecb))
    f.write('spin %i\n' %(neleca-nelecb))
    f.write('use_complex\n')
    
    # I am just keeping this piece of the code:
    if DMRGCIobj.groupname is not None:
        if isinstance(DMRGCIobj.wfnsym, str):
            wfnsym = dmrg_sym.irrep_name2id(DMRGCIobj.groupname, DMRGCIobj.wfnsym)
        else:
            gpname = dmrg_sym.d2h_subgroup(DMRGCIobj.groupname)
            assert(DMRGCIobj.wfnsym in dmrg_sym.IRREP_MAP[gpname])
            wfnsym = DMRGCIobj.wfnsym
        f.write('irrep %i\n' % wfnsym)

    if (not Restart):
        schedule = make_schedule(DMRGCIobj.scheduleSweeps,
                                DMRGCIobj.scheduleMaxMs,
                                DMRGCIobj.scheduleTols,
                                DMRGCIobj.scheduleNoises,
                                DMRGCIobj.twodot_to_onedot)
        f.write('%s\n' % schedule)
    else:
        f.write('schedule\n')
        f.write('0 %6i  %8.4e  %8.4e \n' %(DMRGCIobj.maxM, DMRGCIobj.tol/10, 0e-6))
        f.write('end\n')
        f.write('fullrestart\n')
        f.write('onedot \n')
        if maxIter is None:
            maxIter = 8

    if DMRGCIobj.groupname is not None:
        f.write('sym %s\n' % dmrg_sym.d2h_subgroup(DMRGCIobj.groupname).lower())
    f.write('orbitals %s\n' % DMRGCIobj.integralFile)
    if maxIter is None:
        maxIter = DMRGCIobj.maxIter
    f.write('maxiter %i\n'%maxIter)
    f.write('sweep_tol %8.4e\n'%DMRGCIobj.tol)

    f.write('outputlevel %s\n'%DMRGCIobj.outputlevel)
    f.write('hf_occ %s\n'%DMRGCIobj.hf_occ)
    if(with_2pdm and DMRGCIobj.twopdm):
        f.write('twopdm\n')
    if(DMRGCIobj.nonspinAdapted):
        f.write('nonspinAdapted\n')
    if(DMRGCIobj.scratchDirectory):
        f.write('prefix  %s\n'%DMRGCIobj.scratchDirectory)
    if (DMRGCIobj.nroots !=1):
        f.write('nroots %d\n'%DMRGCIobj.nroots)
        if (DMRGCIobj.weights==[]):
            DMRGCIobj.weights= [1.0/DMRGCIobj.nroots]* DMRGCIobj.nroots
        f.write('weights ')
        for weight in DMRGCIobj.weights:
            f.write('%f '%weight)
        f.write('\n')

    block_extra_keyword = DMRGCIobj.extraline + DMRGCIobj.block_extra_keyword + extraline
    if block_version(DMRGCIobj.executable).startswith('1.1'):
        for line in block_extra_keyword:
            if not ('num_thrds' in line or 'memory' in line):
                f.write('%s\n'%line)
    else:
        if DMRGCIobj.memory is not None:
            f.write('memory, %i, g\n'%(DMRGCIobj.memory))
        if DMRGCIobj.num_thrds > 1:
            f.write('num_thrds %d\n'%DMRGCIobj.num_thrds)
        for line in block_extra_keyword:
            f.write('%s\n'%line)
    f.close()
    return confFile

def writeIntegralFile(DMRGCIobj, h1e, eri, ncas, nelec, ecore=0):
    if isinstance(nelec, (int, np.integer)):
        neleca = nelec//2 + nelec%2
        nelecb = nelec - neleca
    else :
        neleca, nelecb = nelec

    integralFile = os.path.join(DMRGCIobj.runtimeDir, DMRGCIobj.integralFile)
    if DMRGCIobj.groupname is not None and DMRGCIobj.orbsym is not []:
        # This is one last hook to avoid using orbital symmetries.
        raise NotImplementedError("Complex integrals with symmetry is not implemented yet.")

    assert h1e.shape == (ncas, ncas)
    assert eri.shape == (ncas, ncas, ncas, ncas)
    cmd = ' '.join((DMRGCIobj.mpiprefix, "mkdir -p", DMRGCIobj.scratchDirectory))
    check_call(cmd, shell=True)
    if not os.path.exists(DMRGCIobj.runtimeDir):
        os.makedirs(DMRGCIobj.runtimeDir)

    from_integrals(integralFile, h1e, eri, ncas,
                                neleca+nelecb, ecore, ms=abs(neleca-nelecb),
                                orbsym=DMRGCIobj.orbsym)
    return integralFile
    
class DMRGCIComplex(DMRGCI):

    def kernel(self, h1e, eri, norb, nelec, fciRestart=None, ecore=0, **kwargs):
        if self.nroots == 1:
            roots = 0
        else:
            roots = range(self.nroots)
        if fciRestart is None:
            fciRestart = self.restart or self._restart

        if 'orbsym' in kwargs:
            self.orbsym = kwargs['orbsym']
        writeIntegralFile(self, h1e, eri, norb, nelec, ecore)
        writeDMRGConfFile(self, nelec, fciRestart)
        # Adapt spin to match the number of alpha and beta electrons supplied to kernel.
        if lib.isintsequence(nelec):
            neleca, nelecb = nelec
            self.spin = neleca - nelecb
        if self.verbose >= logger.DEBUG1:
            inFile = os.path.join(self.runtimeDir, self.configFile)
            logger.debug1(self, 'Block Input conf')
            logger.debug1(self, open(inFile, 'r').read())
        if self.onlywriteIntegral:
            logger.info(self, 'Only write integral')
            try:
                calc_e = readEnergy(self)
            except IOError:
                if self.nroots == 1:
                    calc_e = 0.0
                else :
                    calc_e = [0.0] * self.nroots
            return calc_e, roots
        if self.returnInt:
            return h1e, eri

        executeBLOCK(self)
        if self.verbose >= logger.DEBUG1:
            outFile = os.path.join(self.runtimeDir, self.outputFile)
            logger.debug1(self, open(outFile).read())
        calc_e = readEnergy(self)
        if self.restart:
            # Restart only the first iteration
            self.restart = False

        return calc_e, roots