# Essentially a deoptimized implementation of LASSCF which is just a
# FCISolver object that does everything in the Fock-space FCI basis
# with no constraint for spin or charge.
#
# Must define:
#   kernel = approx_kernel
#   make_rdm12
#   get_init_guess
#
# In these functions, except for get_init_guess, "nelec" is ignored

import time, math
import numpy as np
from scipy import linalg, optimize
from pyscf import lib
from pyscf.fci import direct_spin1, cistring, spin_op
from itertools import product
from mrh.exploratory.citools import fockspace
from mrh.exploratory.unitary_cc.uccsd_sym1 import get_uccs_op
from mrh.my_pyscf.mcscf.lasci import all_nonredundant_idx
from itertools import product

verbose_lbjfgs = [-1,-1,-1,0,50,99,100,101,101,101]

def kernel (fci, h1, h2, norb, nelec, nlas=None, ci0_f=None,
            tol=1e-8, gtol=1e-4, max_cycle=15000, 
            orbsym=None, wfnsym=None, ecore=0, **kwargs):

    if nlas is None: nlas = getattr (fci, 'nlas', [norb])
    if ci0_f is None: ci0_f = fci.get_init_guess (norb, nelec, nlas, h1, h2)
    verbose = kwargs.get ('verbose', getattr (fci, 'verbose', 0))
    if isinstance (verbose, lib.logger.Logger):
        log = verbose
        verbose = log.verbose
    else:
        log = lib.logger.new_logger (fci, verbose)

    las = fci.get_obj (h1, h2, ci0_f, norb, nlas, nelec, ecore=ecore, log=log)
    las_options = {'gtol':     gtol,
                   'maxiter':  max_cycle,
                   'disp':     verbose>lib.logger.DEBUG}
    log.info ('LASCI object has %d degrees of freedom', las.nvar_tot)
    res = optimize.minimize (las, las.get_init_guess (),
        method='BFGS', jac=True, callback=las.solver_callback,
        options=las_options)
    assert (res.success)

    fci.converged = res.success
    e_tot = las.energy_tot (res.x)
    ci1 = las.get_fcivec (res.x)
    if verbose>=lib.logger.DEBUG:
        las.uop.print_tab (_print_fn=log.debug)
        las.print_x (res.x, _print_fn=log.debug)
    return e_tot, ci1

def make_rdm1 (fci, fcivec, norb, nelec, **kwargs):
    dm1 = np.zeros ((norb, norb))
    for nelec in product (range (norb+1), repeat=2):
        ci = fockspace.fock2hilbert (fcivec, norb, nelec)
        d = direct_spin1.make_rdm1 (ci, norb, nelec, **kwargs)
        dm1 += d
    return dm1

def make_rdm1s (fci, fcivec, norb, nelec, **kwargs):
    dm1a = np.zeros ((norb,norb))
    dm1b = np.zeros ((norb,norb))
    for nelec in product (range (norb+1), repeat=2):
        ci = fockspace.fock2hilbert (fcivec, norb, nelec)
        da, db = direct_spin1.make_rdm1s (ci, norb, nelec, **kwargs)
        dm1a += da
        dm1b += db
    return dm1a, dm1b

def make_rdm12 (fci, fcivec, norb, nelec, **kwargs):
    dm1 = np.zeros ((norb,norb))
    dm2 = np.zeros ((norb,norb,norb,norb))
    for nelec in product (range (norb+1), repeat=2):
        ci = fockspace.fock2hilbert (fcivec, norb, nelec)
        d1, d2 = direct_spin1.make_rdm12 (ci, norb, nelec, **kwargs)
        dm1 += d1
        dm2 += d2
    return dm1, dm2

def get_init_guess (fci, norb, nelec, nlas, h1, h2):
    nelec = direct_spin1._unpack_nelec (nelec)
    hdiag = fci.make_hdiag (h1, h2, norb, nelec)
    ndeta = cistring.num_strings (norb, nelec[0])
    addr_dp = np.divmod (np.argmin (hdiag), ndeta)
    str_dp = [cistring.addr2str (norb, n, c) for n,c in zip (nelec, addr_dp)]
    ci0_f = []
    for n in nlas:
        ndet = 2**n
        c = np.zeros ((ndet, ndet))
        s = [str_dp[0] % ndet, str_dp[1] % ndet]
        c[s[0],s[1]] = 1.0
        str_dp = [str_dp[0] // ndet, str_dp[1] // ndet]
        ci0_f.append (c)
    return ci0_f

def spin_square (fci, fcivec, norb, nelec):
    ss = 0.0
    for ne in product (range (norb+1), repeat=2):
        c = fockspace.fock2hilbert (fcivec, norb, ne)
        ssc = spin_op.contract_ss (c, norb, ne)
        ss += c.conj ().ravel ().dot (ssc.ravel ())
    s = np.sqrt(ss+.25) - .5
    multip = s*2+1
    return ss, multip


class LASCI_ObjectiveFunction (object):
    ''' Evaluate the energy and Jacobian of a LASSCF trial function parameterized in terms
        of unitary CC singles amplitudes and CI transfer operators. '''

    def __init__(self, fcisolver, h1, h2, ci0_f, norb, nlas, nelec, ecore=0, log=None):
        self.fcisolver = fcisolver
        self.h = (ecore, h1, h2)
        self.ci0_f = [ci.copy () for ci in ci0_f]
        self.norb = norb
        self.nlas = nlas = np.asarray (nlas)
        self.nelec = sum (direct_spin1._unpack_nelec (nelec))
        self.nfrags = len (nlas)
        assert (sum (nlas) == norb)
        self.uniq_orb_idx = all_nonredundant_idx (norb, 0, nlas)
        self.nconstr = 1 # Total charge only
        self.log = log if log is not None else lib.logger.new_logger (fcisolver, fcisolver.verbose)
        self.it_cnt = 0
        self.uop = fcisolver.get_uop (norb, nlas)
        self._x_last = np.zeros (self.nvar_tot)
        assert (self.check_ci0_constr ())

    def fermion_spin_shuffle (self, c, norb=None, nlas=None):
        if norb is None: norb = self.norb
        if nlas is None: nlas = self.nlas
        return c * fockspace.fermion_spin_shuffle (norb, nlas)

    def fermion_frag_shuffle (self, c, i, j, norb=None):
        # TODO: fix this!
        if norb is None: norb=self.norb
        c_shape = c.shape
        c = c.ravel ()
        sgn = fockspace.fermion_frag_shuffle (norb, i, j)
        sgn = np.multiply.outer (sgn, sgn).ravel ()
        #if isinstance (sgn, np.ndarray):
        #    print (sgn.shape)
        #    flip_idx = sgn<0
        #    if np.count_nonzero (flip_idx):
        #        c_flip = c.copy ()
        #        c_flip[~flip_idx] = 0.0
        #        det_flp = np.argsort (-np.abs (c_flip))
        #        for det in det_flp[:10]:
        #            deta, detb = divmod (det, 2**norb)
        #            print (fockspace.pretty_str (deta, detb, norb), c_flip[det])
        return (c * sgn).reshape (c_shape)

    def pack (self, xconstr, xcc, xci_f):
        x = [xconstr, xcc]
        ci0_f = self.ci0_f
        for xci, ci0 in zip (xci_f, ci0_f):
            cHx = ci0.conj ().ravel ().dot (xci.ravel ())
            x.append ((xci - (ci0*cHx)).ravel ())
        x = np.concatenate (x)
        return x

    def unpack (self, x):
        xconstr, x = x[:self.nconstr], x[self.nconstr:]

        xcc = x[:self.uop.ngen_uniq] 
        x = x[self.uop.ngen_uniq:]

        xci = []
        for n in self.nlas:
            xci.append (x[:2**(2*n)].reshape (2**n, 2**n))
            x = x[2**(2*n):]

        return xconstr, xcc, xci

    @property
    def nvar_tot (self):
        return self.nconstr + self.uop.ngen_uniq + sum ([c.size for c in self.ci0_f])

    def __call__(self, x):
        c, uc, huc, uhuc, c_f = self.hc_x (x)
        e_tot = self.energy_tot (x, uc=uc, huc=huc)
        jac = self.jac (x, c=c, uc=uc, huc=huc, uhuc=uhuc, c_f=c_f)
        return e_tot, jac

    def energy_tot (self, x, uc=None, huc=None):
        log = self.log
        norm_x = linalg.norm (x)
        t0 = (time.clock (), time.time ())
        if (uc is None) or (huc is None):
            uc, huc = self.hc_x (x)[1:3]
        uc, huc = uc.ravel (), huc.ravel ()
        cu = uc.conj ()
        cuuc = cu.dot (uc)
        cuhuc = cu.dot (huc)
        e_tot = cuhuc/cuuc
        log.timer ('las_obj fn eval', *t0)
        log.debug ('energy value = %f, norm value = %e, |x| = %e', e_tot, cuuc, norm_x)
        if log.verbose > lib.logger.DEBUG: self.check_x_change (x, e_tot0=e_tot)
        return e_tot

    def jac (self, x, c=None, uc=None, huc=None, uhuc=None, c_f=None):
        norm_x = linalg.norm (x)
        log = self.log
        t0 = (time.clock (), time.time ())
        if any ([x is None for x in [c, uc, huc, uhuc, c_f]]):
            c, uc, huc, uhuc, c_f = self.hc_x (x)
        # Revisit the first line below if t ever breaks
        # number symmetry
        jacconstr = self.get_jac_constr (uc)
        t1 = log.timer ('las_obj constr jac', *t0)
        jact1 = self.get_jac_t1 (x, c=c, huc=huc, uhuc=uhuc)
        t1 = log.timer ('las_obj ucc jac', *t1)
        jacci_f = self.get_jac_ci (x, uhuc=uhuc, uci_f=c_f)
        t1 = log.timer ('las_obj ci jac', *t1)
        log.timer ('las_obj jac eval', *t0)
        g = self.pack (jacconstr, jact1, jacci_f)
        norm_g = linalg.norm (g)
        log.debug ('|gradient| = %e, |x| = %e',norm_g, norm_x)
        return g

    def hc_x (self, x):
        xconstr, xcc, xci = self.unpack (x)
        self.uop.set_uniq_amps_(xcc)
        h = self.constr_h (xconstr)
        c_f = self.rotate_ci0 (xci)
        c = self.dp_ci (c_f)
        uc = self.uop (c)
        huc = self.contract_h2 (h, uc)
        uhuc = self.uop (huc, transpose=True)
        return c, uc, huc, uhuc, c_f
        
    def contract_h2 (self, h, ci):
        norb = self.norb
        hci = h[0] * ci
        for neleca, nelecb in product (range (norb+1), repeat=2):
            nelec = (neleca, nelecb)
            h2eff = self.fcisolver.absorb_h1e (h[1], h[2], norb, nelec, 0.5)
            ci_h = np.squeeze (fockspace.fock2hilbert (ci, norb, nelec))
            hc = direct_spin1.contract_2e (h2eff, ci_h, norb, nelec)
            hci += np.squeeze (fockspace.hilbert2fock (hc, norb, nelec))
        return hci
            
    def dp_ci (self, ci_f):
        norb, nlas = self.norb, self.nlas
        ci = np.ones ([1,1], dtype=ci_f[0].dtype)
        for ix, c in enumerate(ci_f):
            ndet = 2**sum(nlas[:ix+1])
            ci = np.multiply.outer (c, ci).transpose (0,2,1,3).reshape (ndet, ndet)
        ci = self.fermion_spin_shuffle (ci, norb=norb, nlas=nlas)
        return ci

    def constr_h (self, xconstr):
        x = xconstr[0]
        h, norb, nelec = self.h, self.norb, self.nelec
        h = [h[0] - (x*nelec), h[1] + (x*np.eye (self.norb)), h[2]]
        return h 

    def rotate_ci0 (self, xci_f):
        ci0, norb = self.ci0_f, self.norb
        ci1 = []
        for dc, c in zip (xci_f, ci0):
            dc -= c * c.conj ().ravel ().dot (dc.ravel ())
            phi = linalg.norm (dc)
            cosp = np.cos (phi)
            if np.abs (phi) > 1e-8: sinp = np.sin (phi) / phi
            else: sinp = 1 # as precise as it can be w/ 64 bits
            ci1.append (cosp*c + sinp*dc)
        return ci1

    def get_jac_constr (self, ci):
        dm1 = self.fcisolver.make_rdm12 (ci, self.norb, 0)[0]
        return np.array ([np.trace (dm1) - self.nelec])

    def get_jac_t1 (self, x, c=None, huc=None, uhuc=None):
        g = []
        xconstr, xcc, xci_f = self.unpack (x)
        self.uop.set_uniq_amps_(xcc)
        if (c is None) or (uhuc is None):
            c, _, _, uhuc = self.hc_x (x)[:4] 
        for duc, uhuc_i in zip (self.uop.gen_deriv1 (c, _full=False), self.uop.gen_partial (uhuc)):
            g.append (2*duc.ravel ().dot (uhuc_i.ravel ()))
        g = self.uop.product_rule_pack (g)
        return np.asarray (g)

    def get_jac_ci (self, x, uhuc=None, uci_f=None):
        # "uhuc": e^-T1 H e^T1 U|ci0>
        # "jacci": Jacobian elements for the CI degrees of freedom
        # subscript_f means a list over fragments
        # subscript_i means this is not a list but it applies to a particular fragment
        xconstr, xcc, xci_f = self.unpack (x)
        if uhuc is None or uci_f is None:
            uhuc, uci_f = self.hc_x (x)[3:]
        uhuc = self.fermion_spin_shuffle (uhuc)
        norb, nlas, ci0_f = self.norb, self.nlas, self.ci0_f
        jacci_f = []
        for ifrag, (ci0, xci) in enumerate (zip (ci0_f, xci_f)):
            uhuc_i = uhuc.copy ()
            # We're going to contract the fragments in ascending order
            # so minor axes disappear first
            # However, we have to skip the ith fragment itself
            norb0, ndet0 = norb, 2**norb
            norb2, ndet2 = 0, 1
            for jfrag, (uci, norb1) in enumerate (zip (uci_f, nlas)):
                norb0 -= norb1
                if (jfrag==ifrag):
                    norb2 = norb1
                    ndet2 = 2**norb1
                    continue
                # norb0, norb1, and norb2 are the number of orbitals in the sectors arranged
                # in major-axis order: the slower-moving orbital indices we haven't touched yet,
                # the orbitals we are integrating over in this particular cycle of the for loop,
                # and the fast-moving orbital indices that we have to keep uncontracted
                # because they correspond to the outer for loop.
                # We want to move the field operators corresponding to the generation of the
                # norb1 set to the front of the operator products in order to integrate
                # with the correct sign.
                uhuc_i = self.fermion_frag_shuffle (uhuc_i, norb2, norb2+norb1, norb=norb0+norb1+norb2)
                ndet0 = 2**norb0
                ndet1 = 2**norb1
                uhuc_i = uhuc_i.reshape (ndet0, ndet1, ndet2, ndet0, ndet1, ndet2)
                uhuc_i = np.tensordot (uhuc_i, uci, axes=((1,4),(0,1))).reshape (ndet0*ndet2, ndet0*ndet2)
            # Given three orthonormal basis states |0>, |p>, and |q>,
            # with U = exp [xp (|p><0| - |0><p|) + xq (|q><0| - |0><q|)],
            # we have @ xq = 0, xp != 0:
            # U|0>      = cos (xp) |0> + sin (xp) |p>
            # dU|0>/dxp = cos (xp) |p> - sin (xp) |0>
            # dU|0>/dxq = sin (xp) |q> / xp
            cuhuc_i = ci0.conj ().ravel ().dot (uhuc_i.ravel ())
            uhuc_i -= ci0 * cuhuc_i # subtract component along |0>
            xp = linalg.norm (xci)
            if xp > 1e-8:
                xci = xci / xp
                puhuc_i = xci.conj ().ravel ().dot (uhuc_i.ravel ())
                uhuc_i -= xci * puhuc_i # subtract component along |p>
                uhuc_i *= math.sin (xp) / xp # evaluate jac along |q>
                dU_dxp  = math.cos (xp) * puhuc_i
                dU_dxp -= math.sin (xp) * cuhuc_i
                uhuc_i += dU_dxp * xci # evaluate jac along |p>
            jacci_f.append (2*uhuc_i)

        return jacci_f

    def solver_callback (self, x):
        it, log = self.it_cnt, self.log
        norm_x = linalg.norm (x)
        norm_g = linalg.norm (self.jac (x))
        e = self.energy_tot (x)
        log.info ('iteration %d, E = %f, |x| = %e, |g| = %e', it, e, norm_x, norm_g)
        if log.verbose >= lib.logger.DEBUG:
            self.check_x_symm (x, e_tot0=e)
        self.it_cnt += 1

    def get_init_guess (self):
        return np.zeros (self.nvar_tot)

    def get_fcivec (self, x):
        xconstr, xcc, xci_f = self.unpack (x)
        uc_f = self.rotate_ci0 (xci_f)
        uc = self.dp_ci (uc_f)
        self.uop.set_uniq_amps_(xcc)
        uc = self.uop (uc)
        return uc / linalg.norm (uc)

    def check_ci0_constr (self):
        norb, nelec = self.norb, self.nelec
        ci0 = self.dp_ci (self.ci0_f)
        neleca_min = max (0, nelec-norb)
        neleca_max = min (norb, nelec)
        w = 0.0
        for neleca in range (neleca_min, neleca_max+1):
            nelecb = nelec - neleca
            c = fockspace.fock2hilbert (ci0, norb, (neleca,nelecb)).ravel ()
            w += c.conj ().dot (c)
        return w>1e-8

    def check_x_change (self, x, e_tot0=None):
        norb, nelec, log = self.norb, self.nelec, self.log
        log.debug ('<x|x_last>/<x|x> = %e', x.dot (self._x_last) / x.dot (x))
        self._x_last = x.copy ()

    def check_x_symm (self, x, e_tot0=None):
        norb, nelec, log = self.norb, self.nelec, self.log
        if e_tot0 is None: e_tot0 = self.energy_tot (x)
        xconstr = self.unpack (x)[0]
        ci1 = self.get_fcivec (x)
        ss = self.fcisolver.spin_square (ci1, norb, nelec)[0]
        n = np.trace (self.fcisolver.make_rdm12 (ci1, norb, nelec)[0])
        h = self.constr_h (xconstr)
        hc1 = self.contract_h2 (self.constr_h (xconstr), ci1).ravel ()
        ci1 = ci1.ravel ()
        cc = ci1.conj ().dot (ci1)
        e_tot1 = ci1.conj ().dot (hc1) / cc
        log.debug ('<Psi|[1,S**2,N]|Psi> = %e, %e, %e ; mu = %e', cc, ss, n, xconstr[0])
        log.debug ('These two energies should be the same: %e - %e = %e',
            e_tot0, e_tot1, e_tot0-e_tot1)

    def print_x (self, x, _print_fn=print, ci_maxlines=10, jac=None):
        norb, nlas = self.norb, self.nlas
        if jac is None: jac = self.jac (x)
        xconstr, xcc, xci_f = self.unpack (x)
        jconstr, jcc, jci_f = self.unpack (jac)
        _print_fn ('xconstr = {}'.format (xconstr))
        #kappa = np.zeros ((norb, norb), dtype=xcc.dtype)
        #kappa[self.uniq_orb_idx] = xcc[:]
        #kappa -= kappa.T
        #umat = linalg.expm (kappa)
        ci1_f = self.rotate_ci0 (xci_f)
        #_print_fn ('umat:')
        #fmt_str = ' '.join (['{:10.7f}',]*norb)
        #for row in umat: _print_fn (fmt_str.format (*row))
        for ix, (xci, ci1, jci, n) in enumerate (zip (xci_f, ci1_f, jci_f, nlas)):
            _print_fn ('Fragment {} x and ci1 leading elements'.format (ix))
            fmt_det = '{:>' + str (max(4,n)) + 's}'
            fmt_str = ' '.join ([fmt_det, '{:>10s}', fmt_det, '{:>10s}', fmt_det, '{:>10s}'])
            _print_fn (fmt_str.format ('xdet', 'xcoeff', 'cdet', 'ccoeff', 'jdet', 'jcoeff'))
            strs_x = np.argsort (-np.abs (xci).ravel ())
            strs_c = np.argsort (-np.abs (ci1).ravel ())
            strs_j = np.argsort (-np.abs (jci).ravel ())
            strsa_x, strsb_x = np.divmod (strs_x, 2**n)
            strsa_c, strsb_c = np.divmod (strs_c, 2**n)
            strsa_j, strsb_j = np.divmod (strs_j, 2**n)
            fmt_str = ' '.join ([fmt_det, '{:10.3e}', fmt_det, '{:10.3e}', fmt_det, '{:10.3e}'])
            for irow, (sa, sb, ca, cb, ja, jb) in enumerate (zip (strsa_x, strsb_x, strsa_c, strsb_c, strsa_j, strsb_j)):
                if irow==ci_maxlines: break
                sdet = fockspace.pretty_str (sa, sb, n)
                cdet = fockspace.pretty_str (ca, cb, n)
                jdet = fockspace.pretty_str (ja, jb, n)
                _print_fn (fmt_str.format (sdet, xci[sa,sb], cdet, ci1[ca,cb], jdet, jci[ja,jb]))

class FCISolver (direct_spin1.FCISolver):
    kernel = kernel
    approx_kernel = kernel
    make_rdm1 = make_rdm1
    make_rdm1s = make_rdm1s
    make_rdm12 = make_rdm12
    get_init_guess = get_init_guess
    spin_square = spin_square
    def get_obj (self, *args, **kwargs):
        return LASCI_ObjectiveFunction (self, *args, **kwargs)
    def get_uop (self, norb, nlas):
        freeze_mask = np.zeros ((norb, norb), dtype=np.bool_)
        for i,j in zip (np.cumsum (nlas)-nlas, np.cumsum(nlas)):
            freeze_mask[i:j,i:j] = True
        return get_uccs_op (norb, freeze_mask=freeze_mask)


