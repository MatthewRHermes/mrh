# !/usr/bin/env python
import sys
import os
import shutil
import re
import contextlib
import numpy as np

from pyscf import lib
from pyscf import gto, scf, mcscf, ao2mo

try:
    import pyblock2
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes
except ImportError:
    raise ImportError("block2 is not installed. Please install pyblock2 with USECOMPLEX=ON to use this code.")

# Integral symm check tolerance
TOL = 1e-10

logger = lib.logger

# Author: Bhavnesh Jangid

# Interface to use the DMRG as the FCI solver.

# Note: For the complex integrals in DMRG-CI, I can not directly use the pyscf+dmrgscf+block2. As it doesn't support complex numbers. However, The block2 is really good in handling various types of Hamiltonians symmetries. I am writing an interface with the SU2 symmetry along with the complex numbers. In the limit of bond dimension going to infinity this would be equivalent to the exact FCI with complex integrals.I have structured most of these functions similar to the ones in pyscf+dmrgscf+block2, so that it would be easier to maintain and update the code in the future.

# Dependency: This code would only require the block2 with USECOMPLEX=ON.


#TODO
# 1. Add the approx_kernel function
# 2. 

@contextlib.contextmanager
def redirect_stdout_stderr_to_file(filename, mode="a"):
    """
    Wrapper to redirect Python and C/C++ stdout/stderr to a file.
    I am doing this separate the block2 DMRG output from the PySCF output and keeping it
    consistent with the dmrgscf + block2 + pyscf structure.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

    sys.stdout.flush()
    sys.stderr.flush()

    old_stdout = os.dup(1)
    old_stderr = os.dup(2)

    try:
        with open(filename, mode) as f:
            os.dup2(f.fileno(), 1)
            os.dup2(f.fileno(), 2)
            yield
            sys.stdout.flush()
            sys.stderr.flush()
    finally:
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        os.close(old_stdout)
        os.close(old_stderr)

class DMRGCICPLX(lib.StreamObject):
    '''
    DMRG-CI wrapper to be used with block2.
    '''
    def __init__(self, scratchDirectory=None, runtimeDir=None, n_threads=8, n_mkl_threads=1, 
                 stack_mem=int(4e9), symm_type=SymmetryTypes.SU2, 
                 bond_dims=(200, 400, 800), noises=(1e-4, 1e-5, 0.0), 
                 thrds=(1e-6, 1e-7, 1e-8), n_sweeps=20, conv_tol=1e-8, clean_scratch=True,
                 verbose=4, iprint=1):
        
        self.verbose = verbose
        self.stdout = None
        self.max_memory = 4000
        if scratchDirectory is None: scratchDirectory = lib.param.TMPDIR
        self.scratchDirectory = scratchDirectory
        self.runtimeDir = runtimeDir if runtimeDir is not None else scratchDirectory
        self.outputFile = "DMRG.log"
        self.integralFile = "FCIDUMP"
        self.n_threads = n_threads
        self.n_mkl_threads = n_mkl_threads
        self.stack_mem = stack_mem
        self.symm_type = symm_type
        self.bond_dims = list(bond_dims)
        self.noises = list(noises)
        self.thrds = list(thrds)
        self.n_sweeps = n_sweeps
        self.conv_tol = conv_tol
        self.clean_scratch = clean_scratch
        self.iprint = iprint

        self.wfnsym = None
        self.orbsym = None
        self.spin = None
        self.nroots = 1

        self.driver = None
        self.mpo = None
        self.ci = None
        self.e_tot = None
        self.norb = None
        self.nelec = None
        self.ecore = None

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        executable = os.path.dirname(pyblock2.__file__)
        runtimeDir = self.runtimeDir
        if runtimeDir is None:
            runtimeDir = self.scratchDirectory
        log.info("")
        log.info("******** Block2 DMRGCI flags ********")
        log.info("executable             = %s", executable)
        log.info("scratchDirectory       = %s", self.scratchDirectory)
        log.info("runtimeDir             = %s", runtimeDir)
        log.info("integralFile           = %s", os.path.join(runtimeDir, self.integralFile))
        log.info("outputFile             = %s", os.path.join(runtimeDir, self.outputFile))

        log.info("maxIter                = %d", self.n_sweeps)
        log.info("scheduleSweeps         = %s", str(list(range(len(self.bond_dims)))))
        log.info("scheduleMaxMs          = %s", str(self.bond_dims))
        log.info("scheduleTols           = %s", str(self.thrds))
        log.info("scheduleNoises         = %s", str(self.noises))
        log.info("symm_type              = %s", str(self.symm_type))
        log.info("stack_mem              = %s", str(self.stack_mem))
        log.info("n_threads              = %d", self.n_threads)
        log.info("n_mkl_threads          = %d", self.n_mkl_threads)
        log.info("conv_tol               = %g", self.conv_tol)
        log.info("maxM                   = %d", max(self.bond_dims))
        log.info("clean_scratch          = %s", str(self.clean_scratch))
        log.info("iprint                 = %s", str(self.iprint))
        log.info("wfnsym                 = %s", self.wfnsym)
        log.info("orbsym                 = %s", self.orbsym)
        log.info("spin                   = %s", self.spin)
        log.info("nroots                 = %s", self.nroots)
        log.info("norb                   = %s", self.norb)
        log.info("nelec                  = %s", str(self.nelec))
        log.info("ecore                  = %s", str(self.ecore))
        log.info("")
        return self

    @staticmethod
    def _unpack_nelec_spin(nelec):
        if isinstance(nelec, tuple):
            nalpha, nbeta = nelec
            return int(nalpha + nbeta), int(nalpha - nbeta)
        n_elec = int(nelec)
        return n_elec, n_elec % 2

    def _make_driver(self, scratch):
        os.makedirs(scratch, exist_ok=True)
        self.driver = DMRGDriver(scratch=scratch, clean_scratch=self.clean_scratch, 
                                 stack_mem=self.stack_mem, symm_type=self.symm_type,
                                 n_threads=self.n_threads, n_mkl_threads=self.n_mkl_threads)
        return self.driver

    def processIntegral(self, h1e, eri, norb):
        '''
        Process the 1e and 2e integrals
        '''
        assert h1e.shape == (norb, norb)
        assert eri.shape == (norb, norb, norb, norb)

        # If complex symmetry:
        if self.symm_type & SymmetryTypes.CPX:
            h1e = h1e.astype(np.complex128)
            eri = eri.astype(np.complex128)

            # check hermiticity
            assert np.linalg.norm(h1e - h1e.conj().T) < TOL, \
                "h1e is not hermitian. Error = {}".format(np.linalg.norm(h1e - h1e.conj().T))
            assert np.linalg.norm(eri - eri.conj().transpose(2, 3, 0, 1)) < TOL, \
                "eri is not symmetric. Error = {}".format(np.linalg.norm(eri -eri.conj().transpose(2, 3, 0, 1)))
            return h1e, eri

        h1e = h1e.astype(np.float64)
        eri = eri.astype(np.float64)
        assert np.linalg.norm(h1e - h1e.T) < TOL, \
            "h1e is not symmetric. Error = {}".format(np.linalg.norm(h1e - h1e.T))
        assert np.linalg.norm(eri - eri.transpose(2, 3, 0, 1)) < TOL, \
            "eri is not symmetric in the last two indices. Error = {}".format(np.linalg.norm(eri - eri.transpose(2, 3, 0, 1)))
        return h1e, eri

    def kernel(self, h1e, eri, norb, nelec, ci0=None, ecore=0.0, **kwargs):
        '''
        Kernel function to perform the DMRG-CI calculation.
        '''
        self.norb = int(norb)
        self.nelec = nelec
        self.ecore = ecore

        n_elec, spin = self._unpack_nelec_spin(nelec)
        self.spin = spin

        h1e, eri = self.processIntegral(h1e, eri, self.norb)

        if os.path.exists(self.scratchDirectory):
            shutil.rmtree(self.scratchDirectory)

        # Create the scratch directory
        os.makedirs(self.scratchDirectory, exist_ok=True)
        kernel_log = os.path.join(self.runtimeDir, self.outputFile)

        # I am using this function to redirect the stdout and stderr of block2 DMRG calculation.
        with redirect_stdout_stderr_to_file(kernel_log, mode="w"):
            # Make the driver
            driver = self._make_driver(self.scratchDirectory)
            # Initialize the system in block2
            driver.initialize_system(n_sites=self.norb, n_elec=n_elec, spin=spin, 
                                     orb_sym=self.orbsym)
            # Prepare the Hamiltonian
            self.mpo = driver.get_qc_mpo( h1e=h1e, g2e=eri, ecore=ecore, iprint=self.iprint)

            # Write the FCIDUMP file for reference
            driver.write_fcidump(h1e, eri, ecore, 
                                 filename=os.path.join(self.runtimeDir, self.integralFile))

            # Get the initial random MPS
            ket = driver.get_random_mps(tag=f"KET",
                                         bond_dim=self.bond_dims[0], nroots=self.nroots, )

            # Perform the DMRG calculation
            energy = driver.dmrg( self.mpo, ket, n_sweeps=self.n_sweeps, bond_dims=self.bond_dims, 
                                 noises=self.noises, thrds=self.thrds, iprint=self.iprint, )

            # In case of multiple roots
            if isinstance(energy, (list, tuple, np.ndarray)):
                energy = energy[0]

            self.e_tot = float(np.real(energy))

            # Save the 2-RDM
            self._save_2rdm(driver, ket, iprint=self.iprint)

        return self.e_tot, self.ci

    def _save_2rdm(self, driver, ket, iprint=0):
        '''
        Save the 2-RDM after the kernel is done. Later on, using the 2-RDM, we can construct
        the 1-RDM and spin-separated 1-RDMs.
        '''
        dm2_b2 = np.asarray(driver.get_2pdm(ket, iprint=iprint))
        rdm2File = os.path.join(self.scratchDirectory, "2pdm.npy")
        dm2 = dm2_b2.transpose(0, 3, 1, 2).copy()
        np.save(rdm2File, dm2)

    def make_rdm2(self, civec, norb, nelec, **kwargs):
        '''
        Make 2-RDM from a given CI vector. Basically read from the 
        saved 2-RDM file.
        '''
        rdm2File = os.path.join(self.scratchDirectory, "2pdm.npy")
        if  not os.path.exists(rdm2File):
            raise RuntimeError("No saved 2-RDM file found.")
        dm2 = np.asarray(np.load(rdm2File, allow_pickle=False))
        assert dm2.ndim == 4 
        assert dm2.shape == (norb, norb, norb, norb)
        return dm2

    def make_rdm1(self, civec, norb, nelec, **kwargs):
        '''
        Make 1-RDM from a given CI vector. Basically, we will use the 2-RDM to reconstruct the 1-RDM.
        '''
        return self.make_rdm12(civec, norb, nelec, **kwargs)[0]

    def make_rdm12(self, civec, norb, nelec, **kwargs):
        '''
        Make 1-RDM and 2-RDM from a given CI vector.
        '''
        neleca, nelecb = _unpack_nele(nelec, self.spin)
        nelectron = neleca + nelecb
        dm2 = self.make_rdm2(civec=civec, norb=norb, nelec=nelec, **kwargs)
        dm1 = np.einsum('pqrr->pq', dm2, optimize=True) / float(nelectron - 1)
        return dm1, dm2

    def make_rdm1s(self, civec, norb, nelec, **kwargs):
        '''
        Spin-separated 1-RDMs. Only implemented for SU2 symmetry currently.
        Basically, we will use the 2-RDM to reconstruct the spin-separated 1-RDMs. The formula is as follows:
        dm1a = 0.5 * (dm1 + dm1n)
        dm1b = 0.5 * (dm1 - dm1n)
        where dm1n = (2 - nelec/2) * dm1 - np.einsum('pkkq->pq', dm2) / (S + 1)
        '''
        neleca, nelecb = _unpack_nele(nelec, self.spin)

        dm1, dm2 = self.make_rdm12(civec=civec, norb=norb, nelec=nelec, **kwargs)
        nelec = neleca + nelecb
        S = 0.5 * (neleca - nelecb)
        
        dm1n = (2 - nelec / 2) * dm1 - np.einsum('pkkq->pq', dm2)
        dm1n *= 1 / (S + 1)
        dm1a, dm1b = (dm1 + dm1n) * .5, (dm1 - dm1n) * .5
        return dm1a, dm1b

    def spin_square(self, civec, norb, nelec):
        '''
        Check the symm type provided. If it is not SU2 then raise an error.
        '''
        symm_type = self.symm_type
        assert bool(re.search(r'\bSU2\b', str(symm_type))) or symm_type, \
            "You are not using spin-adapted symmetry."
        if isinstance(nelec, (int, np.integer)):
                nelecb = nelec//2
                neleca = nelec - nelecb
        else:
            neleca, nelecb = nelec
            s = 0.5 * (neleca - nelecb)
            ss = s * (s+1)
        return ss, s*2+1

def _unpack_nele(nelec, spin):
    if isinstance(nelec, (int, np.integer)):
        nelecb = (nelec-spin) // 2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    return neleca, nelecb

if __name__ == "__main__":
    from pyscf import gto, mcscf, scf

    mol = gto.M(
        atom="""
        C 0 0 0
        C 0 0 1.4
        """,
        basis="6-31G",
        verbose=4,
    )

    mf = scf.RHF(mol).run()

    mc = mcscf.CASSCF(mf, 8, (4,4))
    mc.fcisolver = DMRGCICPLX(
        n_threads=8,
        bond_dims=(200, 500, 1000, 2000),
        noises=(1e-5, 1e-6, 1e-7, 0.0),
        thrds=(1e-8, 1e-9, 1e-10, 1e-11),
        n_sweeps=40,
        clean_scratch=True,
        symm_type=SymmetryTypes.SU2,
        runtimeDir="./dmrg_casscf_test",
    )
    e_casscf = mc.kernel()[0]
    
    mo_coeff = mc.mo_coeff
    rdm1 = mc.fcisolver.make_rdm1(mc.ci, norb=mc.ncas, nelec=mc.nelecas)
    rdm1a, rdm1b = mc.fcisolver.make_rdm1s(mc.ci, norb=mc.ncas, nelec=mc.nelecas)
    rdm1_, rdm2 = mc.fcisolver.make_rdm12(mc.ci, norb=mc.ncas, nelec=mc.nelecas)

    from pyscf.csf_fci import csf_solver
    mc = mcscf.CASCI(mf, 8, (4,4))
    mc.fcisolver = csf_solver(mol, smult=1)
    e_casscf_csf = mc.kernel(mo_coeff)[0]

    rdm1_ref = mc.fcisolver.make_rdm1(mc.ci, norb=mc.ncas, nelec=mc.nelecas)
    rdm2_ref = mc.fcisolver.make_rdm12(mc.ci, norb=mc.ncas, nelec=mc.nelecas)[1]
    rdm1a_ref, rdm1b_ref = mc.fcisolver.make_rdm1s(mc.ci, norb=mc.ncas, nelec=mc.nelecas)

    print("DMRG-CASSCF energy =", e_casscf)
    print("CSF-CASSCF energy =", e_casscf_csf)
    print("Energy difference between DMRG-CASSCF and CSF-CASSCF =", e_casscf - e_casscf_csf)

    print("1-RDM difference between DMRG-CASSCF and CSF-CASSCF =", np.linalg.norm(rdm1 - rdm1_ref))
    print("2-RDM difference between DMRG-CASSCF and CSF-CASSCF =", np.linalg.norm(rdm2 - rdm2_ref))
    print("Spin-separated 1-RDM difference between DMRG-CASSCF and CSF-CASSCF =", np.linalg.norm(rdm1a - rdm1a_ref) + np.linalg.norm(rdm1b - rdm1b_ref))
    print("1-RDM difference between DMRG-CASSCF and CSF-CASSCF =", np.linalg.norm(rdm1_ - rdm1_ref))