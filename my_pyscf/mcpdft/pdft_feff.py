import numpy as np
from scipy import linalg
from pyscf import lib
from pyscf.dft.gen_grid import BLKSIZE
from pyscf.dft.numint import _contract_rho
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
    
    def __init__(self, ot, mo_coeff, ncore, ncas, casdm1, casdm2, max_memory,
            do_cumulant=True):
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
        self.cascm2 = get_2CDM_from_2RDM (casdm2, casdm1)
        self.max_memory = max_memory        
        self.do_cumulant = do_cumulant

        dm1 = 2 * np.eye (nocc, dtype=casdm1.dtype)
        dm1[ncore:,ncore:] = casdm1
        occ_coeff = mo_coeff[:,:nocc]
        no_occ, uno = linalg.eigh (dm1)
        no_coeff = occ_coeff @ uno
        dm1 = occ_coeff @ dm1 @ occ_coeff.conj ().T
        self.dm1 = dm1 = lib.tag_array (dm1, mo_coeff=no_coeff, mo_occ=no_occ)

        self.make_rho = ni._gen_rho_evaluator (ot.mol, dm1, 1)[0]
        
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
        #   nderiv_ao * nao * ngrids            (copying ao)
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
        eot, vot, fot = self.ot.eval_ot (rho, Pi, dderiv=2, weights=weights, 
            _unpack_vot=False)
        vPi = vot[1] if self.do_cumulant else None
        return fot, vPi

    def get_fxot (self, ao, rho0, Pi0, drho, dPi, x, weights, mask):
        fot, vPi = self.get_fot (rho0, Pi0, weights)
        rho1_c, rho1_a, Pi1 = self.make_dens1 (ao, drho, dPi, mask, x)
        fxrho, fxPi = contract_fot (self.ot, fot, rho0, Pi0,
            rho1_c + rho1_a, Pi1)
        Pi1 = fxrho_a = None
        if self.do_cumulant: # fxrho gets the D_all D_c part
                             # the D_c D_a part has to be done separately
            if vPi.ndim == 1: vPi = vPi[None,:]
            fxrho =   _contract_vot_rho (vPi, rho1_c, add_vrho=fxrho)
            fxrho_a = _contract_vot_rho (vPi, rho1_a)
        return fxrho, fxPi, fxrho_a

    def contract_v_ddens (self, v, ddens, ao, weights):
        vw = v * weights[None,:]
        vd = _contract_vot_ao (vw, ddens)
        ao = ao[:vd.shape[0]]
        return np.tensordot (vd, ao, axes=((0,1),(0,1)))

    def __call__ (self, x):
        dg = np.zeros ((self.nocc, self.nao), dtype=x.dtype)
        for ao, mask, weights, coords in self.ni.block_loop (self.ot.mol,
                self.ot.grids, self.nao, self.rho_deriv, self.max_memory,
                blksize=self.get_blocksize ()):
            rho0, Pi0 = self.make_dens0 (ao, mask)
            if ao.ndim == 2: ao = ao[None,:,:]
            drho, dPi = self.make_ddens (ao, rho0, mask)
            fxrho, fxPi, fxrho_a = self.get_fxot (ao, rho0, Pi0, drho, dPi,
                x, weights, mask)
            dg += self.contract_v_ddens (fxrho, drho, ao, weights)
            dg += self.contract_v_ddens (fxPi, dPi, ao, weights)
            if self.do_cumulant: # The D_c D_a part
                drho_c = drho[:,:,:self.ncore]
                dg[:self.ncore] += self.contract_v_ddens (fxrho_a, drho_c,
                    ao, weights)
            
        dg = np.dot (dg, self.mo_coeff)
        return dg

if __name__ == '__main__':
    from pyscf import gto, scf
    from mrh.my_pyscf import mcpdft
    from functools import partial
    mol = gto.M (atom = 'Li 0 0 0; Li 1.2 0 0', basis = '6-31g',
        verbose=lib.logger.DEBUG, output='pdft_veff.log')
    mf = scf.RHF (mol).run ()
    mc = mcpdft.CASSCF (mf, 'tPBE', 2, 2, grids_level=9).run ()
    print ("Ordinary Li2 tPBE energy:",mc.e_tot)

    nao, nmo = mc.mo_coeff.shape
    ncore, ncas, nelecas = mc.ncore, mc.ncas, mc.nelecas
    nocc = ncore+ncas

    from mrh.my_pyscf.mcpdft.orb_scf import *
    casdm1, casdm2 = mc.fcisolver.make_rdm12 (mc.ci, ncas, nelecas)
    g_orb, gorb_update, h_op, h_diag = mc1step_gen_g_hop (mc,
        mc.mo_coeff, 1, casdm1, casdm2, None)
    mc.update_jk_in_ah = partial (mc1step_update_jk_in_ah, mc)
    eot_h_op = EotOrbitalHessianOperator (mc.otfnal, mc.mo_coeff, mc.ncore, 
        mc.ncas, casdm1, casdm2, mc.max_memory, do_cumulant=True)

    print ("g_orb:", linalg.norm (g_orb))
    print ("gorb_update (1,mc.ci):", linalg.norm (gorb_update (1, mc.ci)))
    print ("h_op(0):", linalg.norm (h_op (np.zeros_like (g_orb))))
    print ("eot_h_op(0):", linalg.norm (eot_h_op (np.zeros ((nmo, nmo)))))
    print ("h_diag:", linalg.norm (h_diag))

    x0 = -g_orb/h_diag
    eot_hx = np.zeros ((nmo, nmo), dtype=x0.dtype)
    eot_hx[:nocc,:] = eot_h_op (mc.unpack_uniq_var (x0))
    eot_hx -= eot_hx.T
    eot_hx = mc.pack_uniq_var (eot_hx)
    u0 = mc.update_rotate_matrix (x0)
    print ("\nx0 = -g_orb/h_diag; u0 = expm (x0)")
    print ("g_orb + h_op(x0):", linalg.norm (g_orb + h_op(x0)))
    print ("g_orb + h_op(x0) - eot_h_op(x0):", linalg.norm (g_orb + h_op(x0) - eot_hx))
    print ("gorb_update (u0,mc.ci):", linalg.norm (gorb_update (u0, mc.ci)))

    mc.mo_coeff = np.dot (mc.mo_coeff, u0)
    e_tot, e_ot = mc.energy_tot ()
    print ("Li2 tPBE energy after rotating orbitals:", e_tot)

