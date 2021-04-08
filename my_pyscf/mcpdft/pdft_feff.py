import time
import numpy as np
from scipy import linalg
from pyscf import lib
from pyscf.dft.gen_grid import BLKSIZE
from pyscf.dft.numint import _contract_rho
from pyscf.mcscf import mc1step
from mrh.my_pyscf.mcpdft.otpd import *
from mrh.my_pyscf.mcpdft.otpd import _grid_ao2mo
from mrh.my_pyscf.mcpdft.tfnal_derivs import contract_fot
from mrh.my_pyscf.mcpdft.pdft_veff import _contract_vot_ao, _contract_vot_rho

def _contract_rho_all (bra, ket):
    if bra.ndim == 2: bra = bra[None,:,:]
    if ket.ndim == 2: ket = ket[None,:,:]
    nderiv, ngrids, norb = bra.shape
    rho = np.empty ((nderiv, ngrids), dtype=bra.dtype)
    rho[0] = _contract_rho (bra[0], ket[0])
    for ideriv in range (1,min(nderiv,4)):
        rho[ideriv]  = _contract_rho (bra[ideriv], ket[0])
        rho[ideriv] += _contract_rho (bra[0], ket[ideriv])
    if nderiv > 4: raise NotImplementedError (('ftGGA or Colle-Salvetti type '
        'functionals'))
    return rho

class EotOrbitalHessianOperator (object):
    ''' Callable object for computing
        (f.x)_pq = (int fot * drho/dk_pq drho/dk_rs x_rs dr)
        where fot is the second functional derivative of an on-top energy
        functional, and "rho" is any of the density, the on-top pair density,
        or their derivatives, in the context of an MC-PDFT calculation using
        the ncore, ncas, mo_coeff.

        Does not compute any contribution of the type
        (int vot * d^2 rho / dk_pq dk_rs x_rs dr)
        except optionally for that related to the Pi = (1/4) rho^2 cumulant
        approximation. All other vot terms are more efficiently computed using
        cached effective Hamiltonian tensors that map straightforwardly to
        those involved in CASSCF orbital optimization.
    '''
    
    def __init__(self, mc, ot=None, mo_coeff=None, ncore=None, ncas=None,
            casdm1=None, casdm2=None, max_memory=None, do_cumulant=True,
            incl_d2rho=False):
        if ot is None: ot = mc.otfnal
        if mo_coeff is None: mo_coeff = mc.mo_coeff
        if ncore is None: ncore = mc.ncore
        if ncas is None: ncas = mc.ncas
        if max_memory is None: max_memory = mc.max_memory
        if (casdm1 is None) or (casdm2 is None):
            dm1, dm2 = mc.fcisolver.make_rdm12 (mc.ci, ncas, mc.nelecas)
            if casdm1 is None: casdm1 = dm1
            if casdm2 is None: casdm2 = dm2

        self.ot = ot
        self.verbose, self.stdout = ot.verbose, ot.stdout
        self.ni, self.xctype = ni, xctype = ot._numint, ot.xctype
        self.rho_deriv, self.Pi_deriv = ot.dens_deriv, ot.Pi_deriv
        deriv = ot.dens_deriv
        self.nderiv_rho = (1,4,10)[int (ot.dens_deriv)]
        self.nderiv_Pi = (1,4)[int (ot.Pi_deriv)]
        self.nderiv_ao = (deriv+1)*(deriv+2)*(deriv+3)//6
        self.mo_coeff = mo_coeff
        self.nao, self.nmo = nao, nmo = mo_coeff.shape
        self.ncore = ncore
        self.ncas = ncas
        self.nocc = nocc = ncore + ncas
        self.casdm2 = casdm2
        self.casdm1s = casdm1s = np.stack ([casdm1, casdm1], axis=0)/2
        self.cascm2 = cascm2 = get_2CDM_from_2RDM (casdm2, casdm1)
        self.max_memory = max_memory        
        self.do_cumulant = do_cumulant
        self.incl_d2rho = incl_d2rho

        dm1 = 2 * np.eye (nocc, dtype=casdm1.dtype)
        dm1[ncore:,ncore:] = casdm1
        occ_coeff = mo_coeff[:,:nocc]
        no_occ, uno = linalg.eigh (dm1)
        no_coeff = occ_coeff @ uno
        dm1 = occ_coeff @ dm1 @ occ_coeff.conj ().T
        self.dm1 = dm1 = lib.tag_array (dm1, mo_coeff=no_coeff, mo_occ=no_occ)

        self.make_rho = ni._gen_rho_evaluator (ot.mol, dm1, 1)[0]
        self.pack_uniq_var = mc.pack_uniq_var
        self.unpack_uniq_var = mc.unpack_uniq_var        

        if incl_d2rho: # Include d^2rho/dk^2 type derivatives
            # Also include a full E_OT value and gradient recalculator
            # for debugging purposes
            self.veff1, self.veff2 = mc.get_pdft_veff (mo=mo_coeff,
                casdm1s=casdm1s, casdm2=casdm2, incl_coul=False)
            get_hcore = lambda * args: self.veff1
            with lib.temporary_env (mc, get_hcore=get_hcore):
                g_orb, _, h_op, h_diag = mc1step.gen_g_hop (mc, mo_coeff,
                    np.eye (nmo), casdm1, casdm2, self.veff2)
            # dressed gen_g_hop objects
            from mrh.my_pyscf.mcpdft.orb_scf import get_gorb_update
            gorb_update_u = get_gorb_update (mc, mo_coeff, ncore=ncore,
                ncas=ncas, eot_only=True)
            def delta_gorb (x):
                u = mc.update_rotate_matrix (x)
                g1 = gorb_update_u (u, mc.ci)
                return g1 - g_orb
            jk_null = np.zeros ((ncas,nmo)), np.zeros ((ncore,nmo-ncore))
            update_jk = lambda * args: jk_null
            def d2rho_h_op (x):
                with lib.temporary_env (mc, update_jk_in_ah=update_jk):
                    return h_op (x)
            self.g_orb = g_orb
            self.delta_gorb = delta_gorb
            self.d2rho_h_op = d2rho_h_op
            self.h_diag = h_diag
            # on-top energy and calculator
            mo_cas = mo_coeff[:,ncore:nocc]
            dm1s = np.dot (mo_cas, casdm1s).transpose (1,0,2)
            dm1s = np.dot (dm1s, mo_cas.conj ().T)
            mo_core = mo_coeff[:,:ncore]
            dm1s += (mo_core @ mo_core.conj ().T)[None,:,:]
            dm_list = (dm1s, (casdm1s, (cascm2, None, None)))
            e_ot = mc.energy_dft (ot=ot, mo_coeff=mo_coeff, dm_list=dm_list)
            def delta_eot (x):
                u = mc.update_rotate_matrix (x)
                mo1 = mo_coeff @ u
                mo_cas = mo1[:,ncore:nocc]
                dm1s = np.dot (mo_cas, casdm1s).transpose (1,0,2)
                dm1s = np.dot (dm1s, mo_cas.conj ().T)
                mo_core = mo1[:,:ncore]
                dm1s += (mo_core @ mo_core.conj ().T)[None,:,:]
                dm_list = (dm1s, (casdm1s, (cascm2, None, None)))
                e1 = mc.energy_dft (ot=ot, mo_coeff=mo1, dm_list=dm_list)
                return e1 - e_ot
            self.e_ot = e_ot
            self.delta_eot = delta_eot


    def get_blocksize (self):
        nderiv_ao, nao = self.nderiv_ao, self.nao
        nderiv_rho, nderiv_Pi = self.nderiv_rho, self.nderiv_Pi
        ncas, nocc = self.ncas, self.nocc
        # Ignore everything that doesn't scale with the size of the molecule
        # or the active space
        nvar = 2 + int (self.rho_deriv) + 2*int (self.Pi_deriv)
        ncol = (nderiv_ao*(2*nao+ncas)        # ao + copy + mo_cas
             + (2+nderiv_Pi)*(ncas**2)        # tensor-product intermediate
             + nocc*(2*nderiv_rho+nderiv_Pi)) # drho_a, drho_b, dPi
        ncol *= 1.1 # fudge factor
        ngrids = self.ot.grids.coords.shape[0]
        remaining_floats = (self.max_memory-lib.current_memory()[0]) * 1e6 / 8
        blksize = int (remaining_floats/(ncol*BLKSIZE))*BLKSIZE
        return max(BLKSIZE,min(blksize,ngrids,BLKSIZE*1200))

    def make_dens0 (self, ao, mask):
        rho = self.make_rho (0, ao, mask, self.xctype)
        if ao.ndim == 2: ao = ao[None,:,:]
        rhos = np.stack ([rho, rho], axis=0)/2
        Pi = get_ontop_pair_density (self.ot, rhos, ao, self.casdm1s,
            self.cascm2, self.mo_coeff[:,self.ncore:self.nocc],
            deriv=self.Pi_deriv, non0tab=mask)
        return rho, Pi
        # volatile memory footprint:
        #   nderiv_ao * nao * ngrids            (copying ao in make_rho)
        # + nderiv_ao * ncas * ngrids           (mo_cas on a grid)
        # + (2+nderiv_Pi) * ncas**2 * ngrids    (tensor-product intermediates)

    def make_ddens (self, ao, rho, mask):
        occ_coeff = self.mo_coeff[:,:self.nocc]
        rhos = np.stack ([rho, rho], axis=0)/2
        mo = _grid_ao2mo (self.ot.mol, ao[:self.nderiv_rho], occ_coeff,
            non0tab=mask)
        drhos, dPi = density_orbital_derivative (self.ot, self.ncore, self.ncas,
            self.casdm1s, self.cascm2, rhos, mo, non0tab=mask)
        return drhos.sum (0), dPi
        # persistent memory footprint:
        #   nderiv_rho * nocc * ngrids          (drho)
        # + nderiv_Pi * nocc * ngrids           (dPi)
        # volatile memory footprint:
        #   nderiv_rho * 2 * ngrids             (copying rho)
        # + 2 * nderiv_rho * nocc * ngrids      (mo & copied drho)
        # + (2+nderiv_Pi) * ncas**2 * ngrids    (tensor-product intermediates)

    def make_dens1 (self, ao, drho, dPi, mask, x):
        # the SECOND index of x touches the DENSITY MATRIX
        ngrids = drho.shape[-1]
        ncore, nocc = self.ncore, self.nocc
        occ_coeff_1 = self.mo_coeff @ (x-x.T)[:,:self.nocc]
        mo1 = _grid_ao2mo (self.ot.mol, ao, occ_coeff_1, non0tab=mask)
        Pi1 = _contract_rho_all (mo1[:self.nderiv_Pi], dPi)
        mo1 = mo1[:self.nderiv_rho]
        drho_c, mo1_c = drho[:,:,:ncore],     mo1[:,:,:ncore]
        drho_a, mo1_a = drho[:,:,ncore:nocc], mo1[:,:,ncore:nocc]
        rho1_c = _contract_rho_all (mo1_c, drho_c)
        rho1_a = _contract_rho_all (mo1_a, drho_a)
        return rho1_c, rho1_a, Pi1

    def get_fot (self, rho, Pi, weights):
        rho = np.stack ([rho,rho], axis=0)/2
        eot, vot, fot = self.ot.eval_ot (rho, Pi, dderiv=2, weights=weights)
        return vot, fot

    def get_fxot (self, ao, rho0, Pi0, drho, dPi, x, weights, mask):
        vot, fot = self.get_fot (rho0, Pi0, weights)
        rho1_c, rho1_a, Pi1 = self.make_dens1 (ao, drho, dPi, mask, x)
        rho1 = rho1_c + rho1_a
        fxrho, fxPi = contract_fot (self.ot, fot, rho0, Pi0, rho1, Pi1)
        vrho, vPi = vot
        de = (np.dot (rho1.ravel (), (vrho * weights[None,:]).ravel ())
            + np.dot (Pi1.ravel (), (vPi * weights[None,:]).ravel ()))
        Pi1 = fxrho_a = rho1 = vrho = None
        if self.do_cumulant: # fxrho gets the D_all D_c part
                             # the D_c D_a part has to be done separately
            if vPi.ndim == 1: vPi = vPi[None,:]
            fxrho =   _contract_vot_rho (vPi, rho1_c, add_vrho=fxrho)
            fxrho_a = _contract_vot_rho (vPi, rho1_a)
        return de, fxrho, fxPi, fxrho_a

    def contract_v_ddens (self, v, ddens, ao, weights):
        vw = v * weights[None,:]
        vd = _contract_vot_ao (vw, ddens)
        ao = ao[:vd.shape[0]]
        return np.tensordot (vd, ao, axes=((0,1),(0,1)))

    def __call__ (self, x, packed=False):
        if self.incl_d2rho:
            packed = True
            dg_d2rho = self.d2rho_h_op (x)
        if packed: x = self.unpack_uniq_var (x)
        dg = np.zeros ((self.nocc, self.nao), dtype=x.dtype)
        de = 0
        for ao, mask, weights, coords in self.ni.block_loop (self.ot.mol,
                self.ot.grids, self.nao, self.rho_deriv, self.max_memory,
                blksize=self.get_blocksize ()):
            rho0, Pi0 = self.make_dens0 (ao, mask)
            if ao.ndim == 2: ao = ao[None,:,:]
            drho, dPi = self.make_ddens (ao, rho0, mask)
            dde, fxrho, fxPi, fxrho_a = self.get_fxot (ao, rho0, Pi0, drho, dPi,
                x, weights, mask)
            de += dde
            dg += self.contract_v_ddens (fxrho, drho, ao, weights)
            dg += self.contract_v_ddens (fxPi, dPi, ao, weights)
            if self.do_cumulant: # The D_c D_a part
                drho_c = drho[:,:,:self.ncore]
                dg[:self.ncore] += self.contract_v_ddens (fxrho_a, drho_c,
                    ao, weights)
            
        dg = -np.dot (dg, self.mo_coeff) # I don't understand?
        if packed:
            dg_full = np.zeros ((self.nmo, self.nmo), dtype=dg.dtype)
            dg_full[:self.nocc,:] = dg[:,:]
            dg_full -= dg_full.T
            dg = self.pack_uniq_var (dg_full)
            if self.incl_d2rho: dg += dg_d2rho 
        return dg, de

    def e_de_full (self, x):
        # Compare me to __call__ for debugging purposes
        assert (self.incl_d2rho)
        return self.delta_gorb (x), self.delta_eot (x)


if __name__ == '__main__':
    from pyscf import gto, scf
    from mrh.my_pyscf import mcpdft
    from functools import partial
    mol = gto.M (atom = 'H 0 0 0; H 1.2 0 0', basis = '6-31g',
        verbose=lib.logger.DEBUG, output='pdft_feff.log')
    mf = scf.RHF (mol).run ()
    print (mf.mo_coeff.shape)
    for nelecas, lbl in zip ((2, (2,0)), ('Singlet','Triplet')):
        print (lbl,'case\n')
        for fnal in 'tLDA,VWN3', 'ftLDA,VWN3', 'tPBE':
            mc = mcpdft.CASSCF (mf, fnal, 2, nelecas).run ()
            print ("Ordinary H2 {} energy:".format (fnal),mc.e_tot)

            nao, nmo = mc.mo_coeff.shape
            ncore, ncas, nelecas = mc.ncore, mc.ncas, mc.nelecas
            nocc = ncore+ncas

            from mrh.my_pyscf.mcpdft.orb_scf import *
            casdm1, casdm2 = mc.fcisolver.make_rdm12 (mc.ci, ncas, nelecas)
            g_orb, gorb_update, h_op, h_diag = mc1step_gen_g_hop (mc,
                mc.mo_coeff, 1, casdm1, casdm2, None)
            mc.update_jk_in_ah = partial (mc1step_update_jk_in_ah, mc)
            eot_h_op = EotOrbitalHessianOperator (mc)

            eot_hop = EotOrbitalHessianOperator (mc, incl_d2rho=True)
            print ("g_orb:", linalg.norm (eot_hop.g_orb))
            print ("h_diag:", linalg.norm (eot_hop.h_diag))
            x0 = -eot_hop.g_orb / eot_hop.h_diag
            conv_tab = np.zeros ((8,3), dtype=x0.dtype)
            print ("x0 = g_orb/h_diag:", linalg.norm (x0))
            print (" n " + ' '.join (['{:>10s}',]*6).format ('de_test','de_ref',
                'de_err','dg_test','dg_ref','dg_err'))
            for p in range (10):
                fac = 1/(2**p)
                x1 = x0 * fac
                dg_test, de_test = eot_hop (x1)
                dg_ref,  de_ref  = eot_hop.e_de_full (x1)
                e_err = (de_test-de_ref)/de_ref
                idx = np.argmax (np.abs (dg_test-dg_ref))
                dg_test_max = dg_test[idx]
                dg_ref_max = dg_ref[idx]
                g_err = (dg_test_max-dg_ref_max)/dg_ref_max
                row = [p, de_test, de_ref, e_err, dg_test_max, dg_ref_max, g_err]
                print (("{:2d} " + ' '.join (['{:10.3e}',]*6)).format (*row))
            dg_err = dg_test - dg_ref
            denom = dg_ref.copy ()
            denom[np.abs(dg_ref)<1e-8] = 1.0
            dg_err /= denom
            fmt_str = ' '.join (['{:10.3e}',]*nmo)
            print ("dg_test:")
            for row in mc.unpack_uniq_var (dg_test): print (fmt_str.format (*row))
            print ("dg_ref:")
            for row in mc.unpack_uniq_var (dg_ref): print (fmt_str.format (*row))
            fmt_str = ' '.join (['{:6.2f}',]*nmo)
            print ("dg_err (relative):")
            for row in mc.unpack_uniq_var (dg_err): print (fmt_str.format (*row))
            print ("")
        print ("")

