import numpy as np
import copy
from pyscf.lib import logger
from pyscf.dft.gen_grid import Grids
from pyscf.dft.numint import _NumInt, NumInt
from mrh.util import params
from mrh.my_pyscf.mcpdft import pdft_veff, tfnal_derivs

class otfnal:
    r''' Needs:
        mol: object of class pyscf.gto.mole
        grids: object of class pyscf.dft.gen_grid.Grids
        get_E_ot: function with calling signature shown below
        _numint: object of class pyscf.dft.NumInt
            member functions "hybrid_coeff", "nlc_coeff, "rsh_coeff", and "_xc_type" (at least)
            must be overloaded; see below
        verbose: integer
            for PySCF's logger system
        stdout: record
            for PySCF's logger system
        otxc: string
            name of on-top pair-density exchange-correlation functional
    '''

    def __init__ (self, mol, **kwargs):
        self.mol = mol
        self.verbose = mol.verbose
        self.stdout = mol.stdout    

    Pi_deriv = 0

    def _init_info (self):
        logger.info (self, 'Building %s functional', self.otxc)
        omega, alpha, hyb = self._numint.rsh_and_hybrid_coeff(self.otxc, spin=self.mol.spin)
        if hyb > 0:
            logger.info (self, 'Hybrid functional with %s CASSCF exchange', hyb)

    @property
    def xctype (self):
        return self._numint._xc_type (self.otxc)

    @property
    def dens_deriv (self):
        return ['LDA', 'GGA', 'MGGA'].index (self.xctype)

    def get_E_ot (self, rho, Pi, weight):
        r''' get the on-top energy

        Args:
            rho : ndarray of shape (2,*,ngrids)
                containing spin-density [and derivatives]
            Pi : ndarray with shape (*,ngrids)
                containing on-top pair density [and derivatives]
            weight : ndarray of shape (ngrids)
                containing numerical integration weights

        Returns : float
            The on-top exchange-correlation energy for the given on-top xc functional
        '''

        raise RuntimeError("on-top xc functional not defined")
        return 0

    get_veff_1body = pdft_veff.get_veff_1body

    def get_dEot_drho (self, rho, Pi, **kwargs):
        r''' get the functional derivative dE_ot/drho

        Args:
            rho : ndarray of shape (2,*,ngrids)
                containing spin-density [and derivatives]
            Pi : ndarray with shape (*,ngrids)
                containing on-top pair density [and derivatives]

        Returns: ndarray of shape (2,*,ngrids)
            The functional derivative of the on-top pair density exchange-correlation
            energy wrt to spin density and its derivatives
        '''

        raise RuntimeError("on-top xc functional not defined")
        return 0

    get_veff_2body = pdft_veff.get_veff_2body
    get_veff_2body_kl = pdft_veff.get_veff_2body_kl

    def get_dEot_dPi (self, rho, Pi, **kwargs):
        r''' get the functional derivative dE_ot/dPi

        Args:
            rho : ndarray of shape (2,*,ngrids)
                containing spin-density [and derivatives]
            Pi : ndarray with shape (*,ngrids)
                containing on-top pair density [and derivatives]

        Returns: ndarray of shape (*,ngrids)
            The functional derivative of the on-top pair density exchange-correlation
            energy wrt to the on-top pair density and its derivatives
        '''

        raise RuntimeError("on-top xc functional not defined")
        return 0

class transfnal (otfnal):
    r''' "translated functional" of Li Manni et al., JCTC 10, 3669 (2014).
    '''

    def __init__ (self, ks, **kwargs):
        otfnal.__init__(self, ks.mol, **kwargs)
        self.otxc = 't' + ks.xc
        self._numint = copy.copy (ks._numint)
        self.grids = copy.copy (ks.grids)
        self._numint.hybrid_coeff = t_hybrid_coeff.__get__(self._numint)
        self._numint.nlc_coeff = t_nlc_coeff.__get__(self._numint)
        self._numint.rsh_coeff = t_rsh_coeff.__get__(self._numint)
        self._numint.eval_xc = t_eval_xc.__get__(self._numint)
        self._numint._xc_type = t_xc_type.__get__(self._numint)
        self._init_info ()

    def get_E_ot (self, rho, Pi, weight):
        r''' E_ot[rho, Pi] = V_xc[rho_translated] 
    
            Args:
                rho : ndarray of shape (2,*,ngrids)
                    containing spin-density [and derivatives]
                Pi : ndarray with shape (*,ngrids)
                    containing on-top pair density [and derivatives]
                weight : ndarray of shape (ngrids)
                    containing numerical integration weights
    
            Returns : float
                The on-top exchange-correlation energy, for an on-top xc functional
                which uses a translated density with an otherwise standard xc functional
        '''
        assert (rho.shape[1:] == Pi.shape[:]), "rho.shape={0}, Pi.shape={1}".format (rho.shape, Pi.shape)
        if rho.ndim == 2:
            rho = np.expand_dims (rho, 1)
            Pi = np.expand_dims (Pi, 0)
            
        rho_t = self.get_rho_translated (Pi, rho)
        rho = np.squeeze (rho)
        Pi = np.squeeze (Pi)

        # E_ot[rho,Pi] = \int {dE_ot/ddens}(r) * dens(r) dr
        #              = \sum_i {dE_ot/ddens}_i * dens_i * weight_i
        dexc_ddens  = self._numint.eval_xc (self.otxc, (rho_t[0,:,:], rho_t[1,:,:]), spin=1, relativity=0, deriv=0, verbose=self.verbose)[0]
        rho_t = rho_t[:,0,:].sum (0)
        rho_t *= weight
        dexc_ddens *= rho_t

        return dexc_ddens.sum ()

    def get_ratio (self, Pi, rho_avg):
        r''' R = Pi / [rho/2]^2 = Pi / rho_avg^2
            An intermediate quantity when computing the translated spin densities

            Note this function returns 1 for values and 0 for derivatives for every point where the charge density is close to zero (i.e., convention: 0/0 = 1)
        '''
        assert (Pi.shape == rho_avg.shape)
        nderiv = Pi.shape[0]
        if nderiv > 4:
            raise NotImplementedError("derivatives above order 1")

        R = np.zeros_like (Pi)  
        R[0,:] = 1
        idx = rho_avg[0] >= (1e-15 / 2)
        # Chain rule!
        for ideriv in range (nderiv):
            R[ideriv,idx] = Pi[ideriv,idx] / rho_avg[0,idx] / rho_avg[0,idx]
        # Product rule!
        for ideriv in range (1,nderiv):
            R[ideriv,idx] -= 2 * rho_avg[ideriv,idx] * R[0,idx] / rho_avg[0,idx]
        return R

    def get_rho_translated (self, Pi, rho, Rmax=1, zeta_deriv=False, weights=None):
        r''' original translation, Li Manni et al., JCTC 10, 3669 (2014).
        rho_t[0] = {(rho[0] + rho[1]) / 2} * (1 + zeta)
        rho_t[1] = {(rho[0] + rho[1]) / 2} * (1 - zeta) 
    
        where
    
        zeta = (1-ratio)^(1/2) ; ratio < 1
             = 0               ; otherwise
        with
        ratio = Pi / [{(rho[0] + rho[1]) / 2}^2]
    
            Args:
                Pi : ndarray of shape (*, ngrids)
                    containing on-top pair density [and derivatives]
                rho : ndarray of shape (2, *, ngrids)
                    containing spin density [and derivatives]
    
            Kwargs:
                Rmax : float
                    cutoff for value of ratio in computing zeta; not inclusive
                zeta_deriv : logical
                    whether to include the derivative of zeta in the gradient of rho_t
                weights : ndarray of shape (ngrids)
                    weights for numerical quadrature. Used only to test the integral
                    of rho_t for debugging purposes
    
            Returns: ndarray of shape (2,*,ngrids)
                containing translated spin density (and derivatives)
        '''
        assert (Rmax <= 1), "Don't set Rmax above 1.0!"
        nderiv = rho.shape[1]
        nderiv_zeta = nderiv if zeta_deriv else 1
    
        rho_avg = (rho[0,:,:] + rho[1,:,:]) / 2
        rho_t = rho.copy ()

        R = self.get_ratio (Pi[0:nderiv_zeta,:], rho_avg[0:nderiv_zeta,:])

        # For nonzero charge & pair density, set alpha dens = beta dens = 1/2 charge dens
        idx = (rho_avg[0] >= (1e-15 / 2)) & (Pi[0] >= 1e-15) 
        rho_t[0][:,idx] = rho_t[1][:,idx] = rho_avg[:,idx]

        # For 0 <= ratio < 1 and 0 <= rho, correct spin density using on-top density
        idx &= (Rmax > R[0])
        assert (np.all (R[0,idx] >= 0)), np.amin (R[0,idx])
        assert (np.all (R[0,idx] <= Rmax)), np.amax (R[0,idx])
        zeta = np.empty_like (R[:,idx])
        zeta[0] = np.sqrt (1.0 - R[0,idx])

        # Chain rule!
        for ideriv in range (1, nderiv_zeta):
            zeta[ideriv] = -R[ideriv,idx] / zeta[0] / 2
    
        # Chain rule!
        for ideriv in range (nderiv):
            w = rho_avg[ideriv,idx] * zeta[0]
            rho_t[0,ideriv,idx] += w
            rho_t[1,ideriv,idx] -= w
        # Product rule!
        for ideriv in range (1,nderiv_zeta):
            w = rho_avg[0,idx] * zeta[ideriv]
            rho_t[0,ideriv,idx] += w
            rho_t[1,ideriv,idx] -= w

        if self.verbose > logger.DEBUG and weights is not None:
            nelec = (np.sum (rho_t[:,0,:], axis=0) * weights).sum ()
            lib.logger.debug1 (self, 'Total number of electrons in (this chunk of) the translated density = %s', nelec)

        return rho_t

    eval_ot = tfnal_derivs.eval_ot
    get_bare_vxc = tfnal_derivs.get_bare_vxc
    get_dEot_drho = tfnal_derivs.get_dEot_drho
    get_dEot_dPi = tfnal_derivs.get_dEot_dPi



_FT_R0_DEFAULT=0.9
_FT_R1_DEFAULT=1.15
_FT_A_DEFAULT=-475.60656009
_FT_B_DEFAULT=-379.47331922 
_FT_C_DEFAULT=-85.38149682

class ftransfnal (transfnal):
    r''' "fully translated functional" of Carlson et al., JCTC 11, 4077 (2015)
    '''

    def __init__ (self, ks, **kwargs):
        otfnal.__init__(self, ks.mol, **kwargs)
        self.R0=_FT_R0_DEFAULT
        self.R1=_FT_R1_DEFAULT
        self.A=_FT_A_DEFAULT
        self.B=_FT_B_DEFAULT
        self.C=_FT_C_DEFAULT
        self.otxc = 'ft' + ks.xc
        self._numint = copy.copy (ks._numint)
        self.grids = copy.copy (ks.grids)
        self._numint.hybrid_coeff = ft_hybrid_coeff.__get__(self._numint)
        self._numint.nlc_coeff = ft_nlc_coeff.__get__(self._numint)
        self._numint.rsh_coeff = ft_rsh_coeff.__get__(self._numint)
        self._numint.eval_xc = ft_eval_xc.__get__(self._numint)
        self._numint._xc_type = ft_xc_type.__get__(self._numint)
        self._init_info ()

    Pi_deriv = 1

    def get_rho_translated (self, Pi, rho, Rmax=None, zeta_deriv=True, weights=None):
        r''' "full" translation, Carlson et al., JCTC 11, 4077 (2015)
        rho_t[0] = {(rho[0] + rho[1]) / 2} * (1 + zeta)
        rho_t[1] = {(rho[0] + rho[1]) / 2} * (1 - zeta)
    
        where
        zeta = (1-ratio)^(1/2)                                  ; ratio < R0
           = A*(ratio-R1)^5 + B*(ratio-R1)^4 + C*(ratio-R1)^3 ; R0 <= ratio <= R1
           = 0                                                ; otherwise
    
        Propagate derivatives thru zeta
    
            Args:
                Pi : ndarray of shape (*, ngrids)
                    containing on-top pair density [and derivatives]
                rho : ndarray of shape (2, *, ngrids)
                    containing spin density [and derivatives]
    
            Kwargs:
                Rmax : float
                    cutoff for value of ratio in computing zeta; not inclusive
                zeta_deriv : logical
                    whether to include the derivative of zeta in the gradient of rho_t
                weights : ndarray of shape (ngrids)
                    weights for numerical quadrature. Used only to test the integral

            Returns: ndarray of shape (2,*,ngrids)
                containing fully-translated spin density (and derivatives)
    
        '''
        Rmax = Rmax or self.R1
        nderiv = rho.shape[1]
        if nderiv > 4:
            raise NotImplementedError("derivatives above order 1")
        R0, R1, A, B, C = self.R0, self.R1, self.A, self.B, self.C
    
        rho_ft = super().get_rho_translated (Pi, rho, Rmax=R0, zeta_deriv=True)
        rho_avg = (rho[0] + rho[1]) / 2
        R = self.get_ratio (Pi, rho_avg)
    
        idx = np.where (np.logical_and (R[0] >= R0, R[0] <= R1))[0]
        R_m_R1 = np.stack ([np.power (R[0,idx] - R1, n) for n in range (2,6)], axis=0)
        zeta = np.empty_like (R[:,idx])
        zeta[0] = (A*R_m_R1[5-2] + B*R_m_R1[4-2] + C*R_m_R1[3-2])
        # Chain rule!
        for ideriv in range (1, nderiv):
            zeta[ideriv] = R[ideriv,idx] * (5*A*R_m_R1[4-2] + 4*B*R_m_R1[3-2] + 3*C*R_m_R1[2-2])
    

        # Chain rule!
        for ideriv in range (nderiv):
            rho_ft[0,ideriv,idx] *= (1 + zeta[0])
            rho_ft[1,ideriv,idx] *= (1 - zeta[0])
        # Product rule!
        for ideriv in range (1,nderiv):
            rho_ft[0,ideriv,idx] += rho_avg[0,idx] * zeta[ideriv]
            rho_ft[1,ideriv,idx] -= rho_avg[0,idx] * zeta[ideriv]
    
        if self.verbose > logger.DEBUG and weights is not None:
            nelec = (np.sum (rho_ft[:,0,:], axis=0) * weights).sum ()
            lib.logger.debug1 (self, 'Total number of electrons in (this chunk of) the fully-translated density = %s', nelec)

        return np.squeeze (rho_ft)

_CS_a_DEFAULT = 0.04918
_CS_b_DEFAULT = 0.132
_CS_c_DEFAULT = 0.2533
_CS_d_DEFAULT = 0.349

class colle_salvetti_corr (otfnal):


    def __init__(self, mol, **kwargs):
        super().__init__(mol, **kwargs)
        self.otxc = 'Colle_Salvetti'
        self._numint = NumInt ()
        self.grids = Grids (mol)
        self._numint.hybrid_coeff = lambda * args : 0
        self._numint.nlc_coeff = lambda * args : [0, 0]
        self._numint.rsh_coeff = lambda * args : [0, 0, 0]
        self._numint._xc_type = lambda * args : 'MGGA'
        self.CS_a =_CS_a_DEFAULT
        self.CS_b =_CS_b_DEFAULT
        self.CS_c =_CS_c_DEFAULT
        self.CS_d =_CS_d_DEFAULT 
        self._init_info ()

    def get_E_ot (self, rho, Pi, weight):
        r''' Colle & Salvetti, Theor. Chim. Acta 37, 329 (1975)
        see also Lee, Yang, Parr, Phys. Rev. B 37, 785 (1988) [Eq. (3)]'''

        a, b, c, d = self.CS_a, self.CS_b, self.CS_c, self.CS_d
        rho_tot = rho[0,0] + rho[1,0]
        idx = rho_tot > 1e-15

        num  = -c * np.power (rho_tot[idx], -1/3)
        num  = np.exp (num, num)
        num *= Pi[4,idx]
        num *= b * np.power (rho_tot[idx], -8/3)
        num += 1

        denom  = d * np.power (rho_tot[idx], -1/3)
        denom += 1

        num /= denom
        num *= Pi[0,idx]
        num /= rho_tot[idx]
        num *= weight[idx]

        E_ot  = np.sum (num)
        E_ot *= -4 * a
        return E_ot      
                

def ft_continuity_debug (ot, R, rho, zeta, R0, R1, nrows=50):
    r''' Not working - I need to rethink this '''
    idx = np.argsort (np.abs (R - R0))
    logger.debug (ot, "Close to R0 (%s)", R0)
    logger.debug (ot, "{:19s} {:19s} {:19s} {:19s} {:19s} {:19s} {:19s}".format ("R", "rho_a", "rho_b", "zeta", "zeta_x", "zeta_y", "zeta_z"))
    for irow in idx[:nrows]:
        debugstr = "{:19.12e} {:19.12e} {:19.12e} {:19.12e} {:19.12e} {:19.12e} {:19.12e}".format (R[irow], *rho[:,irow], *zeta[:,irow]) 
        logger.debug (ot, debugstr)
    idx = np.argsort (np.abs (R - R1))
    logger.debug (ot, "Close to R1 (%s)", R1)
    logger.debug (ot, "{:19s} {:19s} {:19s} {:19s} {:19s} {:19s} {:19s}".format ("R", "rho_a", "rho_b", "zeta", "zeta_x", "zeta_y", "zeta_z"))
    for irow in idx[:nrows]:
        debugstr = "{:19.12e} {:19.12e} {:19.12e} {:19.12e} {:19.12e} {:19.12e} {:19.12e}".format (R[irow], *rho[:,irow], *zeta[:,irow]) 
        logger.debug (ot, debugstr)
    


__t_doc__ = "For 'translated' functionals, otxc string = 't' + xc string\n"
__ft_doc__ = "For 'fully translated' functionals, otxc string = 'ft' + xc string\n"

def t_hybrid_coeff(ni, xc_code, spin=0):
    return _NumInt.hybrid_coeff(ni, xc_code[1:], spin=0)
t_hybrid_coeff.__doc__ = __t_doc__ + str(_NumInt.hybrid_coeff.__doc__)

def t_nlc_coeff(ni, xc_code):
    return _NumInt.nlc_coeff(ni, xc_code[1:])
t_nlc_coeff.__doc__ = __t_doc__ + str(_NumInt.nlc_coeff.__doc__)

def t_rsh_coeff(ni, xc_code):
    return _NumInt.rsh_coeff(ni, xc_code[1:])
t_rsh_coeff.__doc__ = __t_doc__ + str(_NumInt.rsh_coeff.__doc__)

def t_eval_xc(ni, xc_code, rho, spin=0, relativity=0, deriv=1, verbose=None):
    return _NumInt.eval_xc(ni, xc_code[1:], rho, spin=spin, relativity=relativity, deriv=deriv, verbose=verbose)
t_eval_xc.__doc__ = __t_doc__ + str(_NumInt.eval_xc.__doc__)

def t_xc_type(ni, xc_code):
    return _NumInt._xc_type(ni, xc_code[1:])
t_xc_type.__doc__ = __t_doc__ + str(_NumInt._xc_type.__doc__)

def t_rsh_and_hybrid_coeff(ni, xc_code, spin=0):
    return _NumInt.rsh_and_hybrid_coeff (ni, xc_code[1:], spin=spin)
t_rsh_and_hybrid_coeff.__doc__ = __t_doc__ + str(_NumInt.rsh_and_hybrid_coeff.__doc__)

def ft_hybrid_coeff(ni, xc_code, spin=0):
    return _NumInt.hybrid_coeff(ni, xc_code[2:], spin=0)
ft_hybrid_coeff.__doc__ = __ft_doc__ + str(_NumInt.hybrid_coeff.__doc__)

def ft_nlc_coeff(ni, xc_code):
    return _NumInt.nlc_coeff(ni, xc_code[2:])
ft_nlc_coeff.__doc__ = __ft_doc__ + str(_NumInt.nlc_coeff.__doc__)

def ft_rsh_coeff(ni, xc_code):
    return _NumInt.rsh_coeff(ni, xc_code[2:])
ft_rsh_coeff.__doc__ = __ft_doc__ + str(_NumInt.rsh_coeff.__doc__)

def ft_eval_xc(ni, xc_code, rho, spin=0, relativity=0, deriv=1, verbose=None):
    return _NumInt.eval_xc(ni, xc_code[2:], rho, spin=spin, relativity=relativity, deriv=deriv, verbose=verbose)
ft_eval_xc.__doc__ = __ft_doc__ + str(_NumInt.eval_xc.__doc__)

def ft_xc_type(ni, xc_code):
    return _NumInt._xc_type(ni, xc_code[2:])
ft_xc_type.__doc__ = __ft_doc__ + str(_NumInt._xc_type.__doc__)

def ft_rsh_and_hybrid_coeff(ni, xc_code, spin=0):
    return _NumInt.rsh_and_hybrid_coeff (ni, xc_code[2:], spin=spin)
ft_rsh_and_hybrid_coeff.__doc__ = __ft_doc__ + str(_NumInt.rsh_and_hybrid_coeff.__doc__)

