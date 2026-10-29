import numpy as np
import ctypes

from pyscf import lib
from pyscf import __config__
from pyscf.fci import direct_uhf
from pyscf.csf_fci.csf import CSFFCISolver as realCSFFCISolver, FCISolver as realFCISolver
from pyscf.csf_fci.csf import unpack_h1e_cs, unpack_h1e_ab, c_arr, get_init_guess, make_hdiag_csf as make_hdiag_csf_real
from pyscf.csf_fci.csf import _debug_g2e as _debug_g2e_real, get_init_guess as get_init_guess_real
from pyscf.lib.numpy_helper import tag_array
from pyscf.csf_fci.csfstring import count_all_csfs

from mrh.my_pyscf.pbc.fci import direct_spin1_cplx
from mrh.my_pyscf.pbc.fci import direct_spin1_cplx_opt
from mrh.my_pyscf.pbc.fci import direct_spin0_cplx
from mrh.my_pyscf.pbc.fci import cplx_csf_helper


# Author: Bhavnesh Jangid

libcsf = lib.load_library('libcsf')

# Note, that the _unpack function should be called from the direct_spin1_cplx, as the 
# _unpack from direct_spin1 will store the link_idx in the tril format.
_unpack = direct_spin1_cplx._unpack
_unpack_nelec = direct_spin1_cplx._unpack_nelec

# Global variables:
HDIAG_IMAG_TOL = 1e-3
IMAG_NOISE = 1e-12

'''
# Okay Great. Let me implement the CSFsolver with complex Hamiltonian.
# Logic of CSFSolvers in PySCF:
#   1. Init guess in det basis, then transform to CSF basis.
#   2. Contruct the Hamiltonian in det basis (using PySCF infrastructure), then transform to CSF basis.
#   3. Solve the eigenvalue problem (exact or Davidson) in CSF basis.
#   4. Transform the eigenvectors back to det basis.
#   5. For the Davidson solver, the matrix-vector product are computed in the det basis and then transformed 
#      to CSF basis for the Davidson solver.

# Now for the complex CI vec of type (a+ib), to use the CSFSolver, the transformation (det to CSF)
# will be a_csf + i*b_csf = U * (a_det + i*b_det), where U is the det to CSF transformation matrix (Note: this
# matrix is not constructed and stored in the memory). The U matrix is real only.
'''

def get_init_guess(norb, nelec, nroots, hdiag_csf, transformer):
    '''
    Also check the doc: csf_fci.csf.get_init_guess
    Get the initial guess for the FCI calculation in the CSF basis.
    Note: for initial guess, I am using the real part of CI vectors only.
    The imaginary part would be small, since the hdiag_csf should be real-dominated.
    args:
        norb: int
            number of active space orbitals
        nelec: tuple (neleca, nelecb) or int
            number of electrons
        nroots: int
            number of roots
        hdiag_csf: np.ndarray of size (1, ncsf)
            diagonal of the Hamiltonian in the CSF basis
        transformer: CSFTransformer object
            the transformer object that can transform between det and CSF basis.
    returns:
        ciout: list of np.ndarray
            list of initial guess CI vectors in the determinant basis.
    '''
    assert np.iscomplexobj(hdiag_csf), "You are using wrong function for real Hamiltonian"
    cireal = get_init_guess_real(norb, nelec, nroots, hdiag_csf.real, transformer)
    ciout = []
    for c in cireal:
        cout = c.astype(hdiag_csf.dtype)
        cout.real = c
        cout.imag = IMAG_NOISE # Only adding some noise.
        cout /= np.linalg.norm(cout) # Normalizing it.
        ciout.append(cout)
    cireal = None
    return ciout

def make_hdiag_det (fci, h1e, eri, norb, nelec):
    r'''
    hdiag = <\psi_I|H_real + i*H_imag|\psi_I> = <\psi_I|H_real|\psi_I> + i*<\psi_I|H_imag|\psi_I>.
    For the Hermitian Hamiltonian, the diagoan elements are real. Still the output array would be
    complex to avoid any datatype bug in any other part of the code.
    args:
        fci: FCISolver object?
            I don't know why it's needed but keeping it consistent with actual csfsolver function.
        h1e: np.ndarray of shape (norb, norb) or (2, norb, norb)
            spin-independent or spin-separated one-electron integrals
        eri: np.ndarray of shape (norb, norb, norb, norb)
            two-electron integrals in chemist's notation
        norb: int
            number of active space orbitals
        nelec: tuple (neleca, nelecb) or int
            number of active space electrons
    returns:
        hdiag: np.ndarray of shape (ndet,)
            diagonal of the Hamiltonian in the determinant basis.
    '''
    assert np.iscomplexobj(h1e) and np.iscomplexobj(eri), \
        "You are using wrong function for real Hamiltonian"
    
    dtype = h1e.dtype
    
    h1ea, h1eb = unpack_h1e_ab(h1e)
    hdiag_real = direct_uhf.make_hdiag(
        [h1ea.real, h1eb.real], 
        [eri.real, eri.real, eri.real], 
        norb, nelec)
    hdiag = hdiag_real.astype(dtype)
    hdiag.real = hdiag_real

    if fci is not None and fci.verbose > lib.logger.DEBUG:
        hdiag_imag  = direct_uhf.make_hdiag(
            [h1ea.imag, h1eb.imag], 
            [eri.imag, eri.imag, eri.imag], 
            norb, nelec)
        if np.abs(hdiag_imag).max() > HDIAG_IMAG_TOL:
            lib.logger.warning("The imaginary part of the Hamiltonian diagonal in the determinant basis "
                               "is not negligible: max imaginary part = %s", np.max(np.abs(hdiag_imag)))
            
    hdiag.imag = 0

    hdiag_real = hdiag_imag = h1ea = h1eb =None
    
    return hdiag

def make_hdiag_csf (h1e, eri, norb, nelec, transformer, hdiag_det=None, max_memory=None, verbose=lib.logger.INFO):
    '''
    Make the diagonal of the Hamiltonian in the CSF basis. Basically, we have the Hamiltonian
    diagonal in the determinant basis (hdiag_det). We will transform it to the CSF basis.
    args:
        h1e: np.ndarray of shape (norb, norb) or (2, norb, norb)
            spin-independent or spin-separated one-electron integrals
        eri: np.ndarray of shape (norb, norb, norb, norb)
            two-electron integrals in chemist's notation
        norb: int
            number of active space orbitals
        nelec: tuple (neleca, nelecb) or int
            number of active space electrons
        transformer: CSFTransformer object
            the transformer object that can transform between det and CSF basis.
        hdiag_det: np.ndarray of shape (ndet,)
            diagonal of the Hamiltonian in the determinant basis.
        max_memory: float
            maximum memory usage.
        verbose: int
            verbosity level.
    returns:
        hdiag_csf: np.ndarray of shape (ncsf,)
            diagonal of the Hamiltonian in the CSF basis.
    '''
    # Constructs the hdiag_det if it is not provided.
    if hdiag_det is None:
        hdiag_det = make_hdiag_det (None, h1e, eri, norb, nelec)

    assert np.iscomplexobj(h1e) and np.iscomplexobj(hdiag_det), \
        "You are using wrong function for real Hamiltonian"
    dtype = h1e.dtype

    hdiag_csf_real = make_hdiag_csf_real(
        h1e.real, eri.real, norb, nelec, transformer, 
        hdiag_det=hdiag_det.real, max_memory=max_memory)
    
    hdiag_csf = hdiag_csf_real.astype(dtype)
    hdiag_csf.real = hdiag_csf_real

    if verbose > lib.logger.DEBUG:
        hdiag_csf_imag = make_hdiag_csf_real(
            h1e.imag, eri.imag, norb, nelec, transformer, 
            hdiag_det=hdiag_det.imag, max_memory=max_memory)
        if np.abs(hdiag_csf_imag).max() > HDIAG_IMAG_TOL:
            lib.logger.warning("The imaginary part of the Hamiltonian diagonal in the CSF basis "
            "is not negligible: max imaginary part = %s", np.amax(np.abs(hdiag_csf_imag)))

    hdiag_csf.imag = 0
    hdiag_csf_real = hdiag_csf_imag = None
    return hdiag_csf

def _debug_g2e (fci, g2e, eri, norb):
    '''
    Debugging 2e part, I am blindly copying the code from csf/csf.py and applying it for the real
    and imag part. Can it be done more efficiently? Probably yes. 
    '''
    _debug_g2e_real (fci, g2e.real, eri.real, norb)
    _debug_g2e_real (fci, g2e.imag, eri.imag, norb)
    return None

def pspace(fci, h1e, eri, norb, nelec, transformer,
           hdiag_det=None, hdiag_csf=None, npsp=200, max_memory=None):
    '''
    In the pspace Davidson solver, we construct a Hamiltonian in smaller subspace 
    and then those eigenvectos are projected to entire subspace and been used for 
    the Davidson solver. 
    The CSF coupling coefficients are real, so the Hamiltonian in the CSF basis is
        H = H_real + 1j * H_imag
    where H_real and H_imag are what the real CSF code (FCICSFpspace_h0tril) gives for
    the real and imaginary parts of the integrals, respectively. H_real is symmetric
    and H_imag is antisymmetric. The matrix is built directly in the CSF basis, so
    the determinant-basis matrix (which scales exponentially with the number of
    unpaired electrons) is never formed.
    args:
        fci: FCISolver object
            the FCI solver object, needed for logging and some other utilities.
        h1e: np.ndarray of shape (norb, norb) or (2, norb, norb)
            spin-independent or spin-separated one-electron integrals
        eri: np.ndarray of shape (norb, norb, norb, norb)
            two-electron integrals in chemist's notation
        norb: int
            number of active space orbitals
        nelec: tuple (neleca, nelecb) or int
            number of active space electrons
        transformer: CSFTransformer object
            the transformer object that can transform between det and CSF basis.
        hdiag_det: np.ndarray of shape (ndet,)
            diagonal of the Hamiltonian in the determinant basis. If not provided, it will be constructed.
        hdiag_csf: np.ndarray of shape (ncsf,)
            diagonal of the Hamiltonian in the CSF basis. If not provided, it will be constructed.
        npsp: int
            number of CSFs in the pspace.
        max_memory: float
            maximum memory usage in MB. If not provided, it will be taken from fci.max_memory.
    returns:
        csf_addr: np.ndarray of shape (npsp,)
            the addresses of the CSFs in the pspace.
        h0: np.ndarray of shape (npsp, npsp)
            the Hamiltonian in the pspace.
    '''
    if norb > 63: raise NotImplementedError('norb > 63')
    if max_memory is None: max_memory=fci.max_memory

    # In complex Hamiltonian, I don't think ao2mo.restore works,
    # so I always keep the eri in the 4D format. 
    assert (h1e.dtype == eri.dtype), \
        "h1e and eri must have the same dtype"
    assert h1e.shape in ((norb, norb), (2, norb, norb)), \
        "h1e must have shape (norb, norb) or (2, norb, norb)"
    assert eri.shape == (norb, norb, norb, norb), \
        "eri must be a 4D array of shape (norb, norb, norb, norb)"

    t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
    neleca, nelecb = _unpack_nelec(nelec)
    h1e = np.ascontiguousarray(h1e)

    # Compute the diagonal elements in both the determinant and CSF basis.
    if hdiag_det is None:
        hdiag_det = fci.make_hdiag(h1e, eri, norb, nelec)
    if hdiag_csf is None:
        hdiag_csf = fci.make_hdiag_csf(h1e, eri, norb, nelec, 
                                       hdiag_det=hdiag_det, max_memory=max_memory, 
                                       verbose=fci.verbose)
    
    assert hdiag_csf.dtype == hdiag_det.dtype == h1e.dtype

    # For Hermitian Hamiltonians, diagonals elements are dominated by the 
    # real-part, that's why I am using only real-part for ranking/selection.
    hdiag_csf_real = hdiag_csf.real
    csf_addr = np.arange(hdiag_csf_real.size, dtype=np.int32)
    
    if transformer.wfnsym is None:
        ncsf_sym = hdiag_csf_real.size
    else:
        idx_sym = transformer.confsym[transformer.econf_csf_mask] == transformer.wfnsym
        ncsf_sym = np.count_nonzero(idx_sym)
        csf_addr = csf_addr[idx_sym]

    if ncsf_sym > npsp:
        try:
            csf_addr = csf_addr[np.argpartition(hdiag_csf_real[csf_addr], npsp - 1)[:npsp]]
        except AttributeError:
            csf_addr = csf_addr[np.argsort(hdiag_csf_real[csf_addr])[:npsp]]

    npsp_csf = len(csf_addr)
    lib.logger.debug(fci, "csf.pspace: building %s-CSF pspace directly in the CSF basis", npsp_csf)

    dconfstrs, sconfstrs, spinstrs = transformer.csfaddrs2str(csf_addr)
    dconfstrs[dconfstrs < 0] = 0
    spin = neleca - nelecb
    h1e_c, h1e_s = unpack_h1e_cs(h1e)

    # Check the eri for nans/infs.
    _debug_g2e(fci, eri, eri, norb)
    t0 = lib.logger.timer_debug1(fci, "csf.pspace: index manipulation", *t0)

    def _build_h0_upper(h1e_c, h1e_s, g2e):
        '''Real CSF-basis matrix elements for real integrals. Only the upper triangle is filled.'''
        h0 = np.zeros((npsp_csf, npsp_csf), dtype=np.float64)
        h1e_c = np.ascontiguousarray(h1e_c, dtype=np.float64)
        h1e_s = np.ascontiguousarray(h1e_s, dtype=np.float64)
        g2e = np.ascontiguousarray(g2e, dtype=np.float64)
        libcsf.FCICSFpspace_h0tril(
            c_arr(h0), c_arr(h1e_c), c_arr(h1e_s), c_arr(g2e),
            c_arr(dconfstrs), c_arr(sconfstrs), c_arr(spinstrs),
            ctypes.c_size_t(npsp_csf), ctypes.c_uint(norb), ctypes.c_int(spin))
        return h0

    h0real = _build_h0_upper(h1e_c.real, h1e_s.real, eri.real)
    h0real += h0real.T
    h0imag = _build_h0_upper(h1e_c.imag, h1e_s.imag, eri.imag)
    # The C routine stores <bra|H|ket> in the transposed slot [ket,bra], so the lower
    # triangle of the Hermitian H_imag is the transposed upper triangle, with the sign
    # flipped: H_imag is antisymmetric with a zero diagonal.
    h0imag = h0imag.T - h0imag

    h0 = h0real.astype(h1e.dtype)
    h0.imag = h0imag
    h0[np.diag_indices_from(h0)] = hdiag_csf[csf_addr].real
    h0real = h0imag = None

    t0 = lib.logger.timer_debug1(fci, "csf.pspace wrapup", *t0)
    return csf_addr, h0

def kernel(fci, h1e, eri, norb, nelec, smult=None, idx_sym=None, ci0=None,
           tol=None, lindep=None, max_cycle=None, max_space=None,
           nroots=None, davidson_only=None, pspace_size=None, max_memory=None,
           orbsym=None, wfnsym=None, ecore=0, transformer=None, **kwargs):
    '''
    kernel function.
    '''
    t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
    if 'verbose' in kwargs:
        verbose = kwargs['verbose']
        kwargs.pop ('verbose')
    else:
        verbose = lib.logger.Logger (stdout=fci.stdout, verbose=fci.verbose)
    
    # I think we should do the sanity check always:
    fci.check_sanity()

    if nroots is None: nroots = fci.nroots
    if pspace_size is None: pspace_size = fci.pspace_size
    if davidson_only is None: davidson_only = fci.davidson_only
    if transformer is None: transformer = fci.transformer
    if max_memory is None: max_memory = fci.max_memory

    nelec = neleca, nelecb = _unpack_nelec(nelec, fci.spin)
    t0 = lib.logger.timer_debug1 (fci, "csf.kernel: throat-clearing", *t0)
    hdiag_det = fci.make_hdiag (h1e, eri, norb, nelec)
    t0 = lib.logger.timer_debug1 (fci, "csf.kernel: hdiag_det", *t0)
    hdiag_csf = fci.make_hdiag_csf (h1e, eri, norb, nelec, hdiag_det=hdiag_det, 
                                    max_memory=max_memory)
    t0 = lib.logger.timer_debug1 (fci, "csf.kernel: hdiag_csf", *t0)
    
    ncsf_all = count_all_csfs (norb, neleca, nelecb, smult)
    if idx_sym is None: ncsf_sym = ncsf_all
    else: ncsf_sym = np.count_nonzero (idx_sym)
    nroots = min(ncsf_sym, nroots)
    if nroots is not None:
        assert (ncsf_sym >= nroots), "Can't find {} roots among only {} CSFs".format (nroots, ncsf_sym)
    
    # Remember, this _unpack should be used from direct_spin1_cplx.
    link_indexa, link_indexb = _unpack(norb, nelec, None)

    na = link_indexa.shape[0]
    nb = link_indexb.shape[0]
    t0 = lib.logger.timer_debug1 (fci, "csf.kernel: throat-clearing", *t0)
    addr, h0 = fci.pspace(h1e, eri, norb, nelec, idx_sym=idx_sym, hdiag_det=hdiag_det, hdiag_csf=hdiag_csf,
                        npsp=max(pspace_size,nroots))
    lib.logger.debug1 (fci, 'csf.kernel: error of hdiag_csf: %s', np.amax (np.abs (hdiag_csf[addr]-np.diag (h0))))
    t0 = lib.logger.timer_debug1 (fci, "csf.kernel: make pspace", *t0)

    # Making sure h0 is Hermitian.
    h0 = 0.5 * (h0 + h0.conj().T)

    if pspace_size > 0:
        pw, pv = fci.eig(h0)
    else:
        pw = pv = None

    dtype = h1e.dtype
    if pspace_size >= ncsf_sym and not davidson_only:
        if ncsf_sym == 1:
            civec = cplx_csf_helper.vec_csf2det_cplx(
                transformer, pv[:,0].reshape(1, 1), normalize=True
            )
            # Parent real CSF solver does not reshape the output CI.
            # I think that might be a bug, but I will keep it consistent with 
            # the parent solver.
            return pw[0]+ecore, civec #.reshape(na, nb)
        elif nroots > 1:
            civecout = []
            for i in range(nroots):
                civeccsf = pv[:,i].T # Should I take the conj here?: I think no.
                civec = cplx_csf_helper.vec_csf2det_cplx(
                    transformer, civeccsf, normalize=True
                )
                civecout.append(civec.reshape(na,nb))
            return pw[:nroots]+ecore, civecout
        elif abs(pw[0]-pw[1]) > 1e-12:
            civeccsf = np.empty((ncsf_sym), dtype=dtype)
            civeccsf[:] = pv[:,0]
            civec = cplx_csf_helper.vec_csf2det_cplx(
                transformer, civeccsf, normalize=True
            )
            return pw[0]+ecore, civec.reshape(na,nb)
        # For the degenerate single-root case, don't return from here. Same as
        # the real CSF solver, let it continue to the Davidson solver.

    t0 = lib.logger.timer_debug1 (fci, "csf.kernel: throat-clearing", *t0)
    if idx_sym is None:
        precond = fci.make_precond(hdiag_csf, pw, pv, addr)
    else:
        addr_bool = np.zeros (ncsf_all, dtype=np.bool_)
        addr_bool[addr] = True
        precond = fci.make_precond(hdiag_csf[idx_sym], pw, pv, addr_bool[idx_sym])

    t0 = lib.logger.timer_debug1 (fci, "csf.kernel: make preconditioner", *t0)
    h2e = fci.absorb_h1e(h1e, eri, norb, nelec, 0.5)
    t0 = lib.logger.timer_debug1 (fci, "csf.kernel: h2e", *t0)


    def hop(x):
        x_det = cplx_csf_helper.vec_csf2det_cplx(
            transformer, x, normalize=False
        )
        hx = fci.contract_2e(h2e, x_det, norb, nelec, (link_indexa, link_indexb))
        hx_out = cplx_csf_helper.vec_det2csf_cplx(
            transformer, hx, normalize=False
        )
        return hx_out.ravel()

    t0 = lib.logger.timer_debug1 (fci, "csf.kernel: make hop", *t0)

    if ci0 is None:
        if hasattr(fci, 'get_init_guess'):
            def ci0 ():
                ci0_det = fci.get_init_guess(norb, nelec, nroots, hdiag_csf)
                dtype = ci0_det[0].dtype
                ci0_csfout = []
                for c in ci0_det:
                    ci0_csfreal = transformer.vec_det2csf (c.real, normalize=False)
                    ci0_csf = ci0_csfreal.astype(dtype)
                    ci0_csf.real = ci0_csfreal
                    ci0_csf.imag = IMAG_NOISE
                    ci0_csf /= np.linalg.norm(ci0_csf)
                    ci0_csfout.append(ci0_csf)
                return ci0_csfout
        else:
            def ci0():
                x0 = []
                for i in range(nroots):
                    x = np.zeros(ncsf_sym, dtype=h1e.dtype)
                    x[addr[i]] = 1.0 + IMAG_NOISE
                    x0.append(x)
                return x0
    else:
        if isinstance(ci0, np.ndarray) and ci0.size == na*nb:
            ci0_out = cplx_csf_helper.vec_det2csf_cplx(
                transformer, ci0.ravel(), normalize=True
            )
            ci0 = [ci0_out]
        else:
            nrow = len (ci0)
            ci0 = np.asarray(ci0).reshape(nrow, -1, order='C')
            ci0 = cplx_csf_helper.vec_det2csf_cplx(
                transformer, ci0, normalize=True
            )
            ci0 = [c for c in np.asarray(ci0).reshape(nrow, -1)]
    
    t0 = lib.logger.timer_debug1 (fci, "csf.kernel: ci0 handling", *t0)

    if tol is None: tol = fci.conv_tol
    if lindep is None: lindep = fci.lindep
    if max_cycle is None: max_cycle = fci.max_cycle
    if max_space is None: max_space = fci.max_space
    tol_residual = getattr(fci, 'conv_tol_residual', None)

    e, c = fci.eig(hop, ci0, precond, tol=tol, lindep=lindep,
                    max_cycle=max_cycle, max_space=max_space, nroots=nroots,
                    max_memory=max_memory, verbose=verbose, follow_state=True,
                    tol_residual=tol_residual, **kwargs)
    
    dtype = c[0].dtype

    t0 = lib.logger.timer_debug1 (fci, "csf.kernel: running fci.eig", *t0)
    if nroots > 1:
        cout = []
        for ciroot in c:
            croot = cplx_csf_helper.vec_csf2det_cplx(
                transformer, ciroot, order='C', normalize=True
            )
            cout.append(croot)
    else:
        cout = cplx_csf_helper.vec_csf2det_cplx(
            transformer, c, order='C', normalize=True
        )
        
    t0 = lib.logger.timer_debug1 (fci, "csf.kernel: transforming final ci vector", *t0)
    if nroots > 1:
        return e+ecore, [ci.reshape(na,nb) for ci in cout]
    else:
        return e+ecore, cout.reshape(na,nb)
    
class cplxCSFFCISolver:
    '''
    Parent class for the complex FCI solver in CSF basis. This class will implement the 
    necessary functions. 
    # Borrowing functions from the real CSF solver. Only modifying the functions
    # which are directly needed.
    # This class won't be of any use for standalone.
    '''
    _keys = {'smult', 'transformer', 'cell'}
    pspace_size = getattr(__config__, 'fci_csf_FCI_pspace_size', 200)
    make_hdiag = make_hdiag_det

    def __init__ (self, cell, smult, **args):
        self.cell = cell
        self.smult = smult
        self.transformer = None
        self.stdout = cell.stdout
        super().__init__ (**args)
        # FCIBase.__init__ (mol=None) resets max_memory to the pyscf default, so this
        # must come after super().__init__ for the cell's max_memory to take effect.
        self.max_memory = cell.max_memory

    def make_hdiag_csf(self, h1e, eri, norb, nelec, hdiag_det=None, smult=None, max_memory=None):
        self.norb = norb
        self.nelec = nelec
        if smult is not None:
            self.smult = smult
        self.check_transformer_cache ()
        max_memory = max_memory if max_memory is not None else self.max_memory
        return make_hdiag_csf(h1e, eri, norb, nelec, self.transformer, hdiag_det=hdiag_det, 
                                   max_memory=max_memory)

    def absorb_h1e(self, h1e, eri, norb, nelec, fac=1):
        h1e_c, h1e_s = unpack_h1e_cs(h1e)
        h2eff = super().absorb_h1e(h1e_c, eri, norb, nelec, fac=fac)
        if h1e_s is not None:
            h2eff = tag_array(h2eff, h1e_s=h1e_s)
        return h2eff
    
    log_transformer_cache = realCSFFCISolver.log_transformer_cache
    print_transformer_cache = realCSFFCISolver.print_transformer_cache

    def contract_2e(self, eris, fcivec, norb, nelec, link_index=None, **kwargs):
        hc = super().contract_2e(eris, fcivec, norb, nelec, 
                                 link_index=link_index, **kwargs)
        if hasattr(eris, 'h1e_s'):
            hc_s = cplx_csf_helper.contract_1e([eris.h1e_s, -eris.h1e_s],
                                               fcivec, norb, nelec, link_index,)
            hc += hc_s.ravel() if hc.ndim == 1 else hc_s.reshape(hc.shape)
        return hc
    
    def pspace (self, h1e, eri, norb, nelec, hdiag_det=None, hdiag_csf=None, npsp=200, **kwargs):
        self.norb = norb
        self.nelec = nelec
        self.smult = kwargs.pop('smult', self.smult)
        self.check_transformer_cache ()
        max_memory = kwargs.get ('max_memory', self.max_memory)
        return pspace (self, h1e, eri, norb, nelec, self.transformer, hdiag_det=hdiag_det,
                       hdiag_csf=hdiag_csf, npsp=npsp, max_memory=max_memory)


class ComplexCSFFCIKernelMixin:
    '''
    Shared CSF wrapper logic for complex FCI solvers.
    '''

    def get_init_guess(self, norb, nelec, nroots, hdiag_csf, **kwargs):
        '''
        Get the initial guess for the FCI calculation in the CSF basis.
        '''
        self.norb = norb
        self.nelec = nelec
        self.check_transformer_cache()
        return get_init_guess(norb, nelec, nroots, hdiag_csf, self.transformer)

    def kernel(self, h1e, eri, norb, nelec, ci0=None, **kwargs):
        self.norb = norb
        self.nelec = nelec
        self.smult = kwargs.pop('smult', self.smult)

        self.check_transformer_cache()
        self.log_transformer_cache(lib.logger.DEBUG)

        e, c = kernel(
            self, h1e, eri, norb, nelec,
            smult=self.smult,
            idx_sym=None,
            ci0=ci0,
            transformer=self.transformer,
            **kwargs
        )

        self.eci, self.ci = e, c
        return e, c

    check_transformer_cache = realFCISolver.check_transformer_cache


class FCISolver(
    ComplexCSFFCIKernelMixin,
    cplxCSFFCISolver,
    direct_spin1_cplx_opt.FCISolver,):
    '''
    direct_spin1 FCI in CSF basis.
    '''
    pass

class FCISolverSpin0(
    ComplexCSFFCIKernelMixin,
    cplxCSFFCISolver,
    direct_spin0_cplx.FCISolver,
):
    '''
    direct_spin0 FCI in CSF basis.
    '''
    pass
