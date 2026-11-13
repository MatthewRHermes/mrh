# Moved to PySCF-Forge in the LASSCF migration (see MIGRATION.md). The LASSCF Hessian debuggers stay here.
# This module re-exports all of pyscf.mcscf.las.addons (including private names) so that existing
# imports keep working.
import pyscf.mcscf.las.addons as _forge
globals ().update ({k: v for k, v in vars (_forge).items () if not k.startswith ('__')})

def debug_lasscf_hessian_(las, check_horb_matvec=False, perfect_orbital_preconditioner=False):
    ''' Monkeypatch function to debug the LASSCF hessian operator for the synchronous algorithm.
    Computes the full dense orbital-rotation Hessian and outputs its smallest eigenvalues and
    condition number at the beginning of every macrocycle of the kernel. Unless the
    perfect_orbital_preconditioner kwarg is True, this function only adds information to the
    las.stdout stream (at substantial additional computational cost); it does not change the
    execution of the calculation. Output is added at the "info" level of verbosity, so if
    las.verbose is smaller than lib.logger.INFO, no additional output is produced. This is too
    expensive to be the default for verbose = lib.logger.DEBUG, but the higher verbosities do way
    too much.

    Args:
        las : instance of :class:`LASSCFNoSymm`
            The method object to debug. Modified in-place!

    Kwargs:
        check_horb_matvec : logical
            If True, outputs the difference between a direct matrix-vector product with the dense
            Horb and the sparse Horb in every microcycle, in addition to the analysis of the dense
            Horb in every macrocycle. Note the CI sector is ignored on both the internal and
            external indices.
        perfect_orbital_preconditioner : logical
            If True, the orbital sector of the preconditioner function is replaced with direct
            solution for z of Horb.z = x. Note that this leaves the CI sector unchanged, so the
            inner CG iteration will still take multiple cycles to converge in general.

    Returns:
        las : instance of :class:`LASSCFNoSymm`
            Same as arg, after in-place modification.
        parent_hop : class
            The original value of the overwritten las._hop. Reassign it (las._hop = parent_hop) to
            undo the effects of this function.
    '''
    from mrh.my_dmet.orbital_hessian import HessianCalculator
    from mrh.util.la import vector_error
    from pyscf.lib import current_memory
    import os, sys
    parent_hop = las.__class__._hop

    nao = las.mol.nao_nr ()
    max_memory = (las.max_memory - current_memory ()[0])
    reqd_memory = (nao**4)*8*2.5/1e6 # 2.5 is safety margin
    if reqd_memory > max_memory:
        logger.warn (las, ('Insufficient memory (%f required; %f available) to debug Hessian! '
                           'Aborting debug monkeypatch...'), reqd_memory, max_memory)
        return las, parent_hop

    # HessianCalculator uses "print" statements, which sucks. Make them go away.
    class SuppressPrint ():
        def __enter__(self):
            self._true_stdout = sys.stdout
            sys.stdout = open (os.devnull, 'w')
        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout.close ()
            sys.stdout = self._true_stdout

    class _hop (las._hop):
        def __init__(self, *args, **kwargs):
            parent_hop.__init__(self, *args, **kwargs)
            self.Horb_full = Horb_full = self._get_dense_Horb ()
            Horb_evals, Horb_evecs = linalg.eigh (Horb_full)
            Horb_mineval = Horb_evals[np.argmin (Horb_evals)]
            Horb_amineval = Horb_evals[np.argmin (np.abs (Horb_evals))]
            log = logger.new_logger (self.las, self.las.verbose)
            log.info ('Orbital Hessian smallest eigenvalue: %.15g', Horb_mineval)
            log.info ('Orbital Hessian smallest-magnitude eigenvalue: %.15g', Horb_amineval)
            log.info ('Orbital Hessian condition number: %.15e', np.linalg.cond (Horb_full))

        def _get_dense_Horb (self):
            ncore, nocc = self.ncore, self.nocc
            dm1s, mo = self.dm1s, self.mo_coeff
            dm1s_ao = np.dot (mo.conj (), np.dot (dm1s, mo.T)).transpose (1,0,2)
            mo_cas = mo[:,ncore:nocc]
            with SuppressPrint ():
                Horb_full = HessianCalculator (self.las._scf, dm1s_ao, self.cascm2, mo_cas)
                idx = self.ugg.uniq_orb_idx.ravel ()
                Horb_full = Horb_full (mo).reshape (self.nmo**2, self.nmo**2)
            Horb_full = Horb_full[np.ix_(idx,idx)]
            return Horb_full

        if check_horb_matvec:
            def _matvec (self, x):
                # Double-check dense Hessian
                # Only checks the orbital-orbital sector
                nvar_orb = self.ugg.nvar_orb
                Hx_orb_test = self._get_dense_Horb () @ x[:nvar_orb]
                xp = x.copy ()
                xp[nvar_orb:] = 0.0
                Hx_orb_ref = parent_hop._matvec (self, xp)[:nvar_orb]
                log = logger.new_logger (self.las, self.las.verbose)
                err_norm, err_angle = vector_error (Hx_orb_test, Hx_orb_ref)
                log.info ('|Horbx - Horbx_ref| = %.5g, %.5g', err_norm, err_angle)
                return parent_hop._matvec (self, x)

        if perfect_orbital_preconditioner:
            def get_prec (self):
                Hci_diag = np.concatenate (self._get_Hci_diag ())
                Hci_diag += self.ah_level_shift
                Hci_diag[np.abs (Hci_diag)<1e-8] = 1e-8
                def prec_op (x):
                    xorb, xci = x[:self.ugg.nvar_orb], x[self.ugg.nvar_orb:]
                    Mxorb = linalg.solve (self.Horb_full, xorb)
                    Mxci = xci / Hci_diag
                    return np.append (Mxorb, Mxci)
                return LinearOperator (self.shape, matvec=prec_op, dtype=self.dtype)
    las._hop = _hop
    return las, parent_hop

class lasscf_hessian_debugger (object):
    ''' Context-manager version of debug_lasscf_hessian_.

    debug_lasscf_hessian_.__doc__:

    ''' + debug_lasscf_hessian_.__doc__
    def __init__(self, las, **kwargs):
        self.las = las
        self.old_hop = None
        self.kwargs = kwargs
    def __enter__(self):
        self.las, self.old_hop = debug_lasscf_hessian_(self.las, **self.kwargs)
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.las._hop = self.old_hop
