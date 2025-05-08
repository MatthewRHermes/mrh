from functools import reduce
import numpy as np
import scipy
from pyscf import lib, mcscf, mcpdft
from pyscf.mcpdft import _dms
from pyscf.lib import logger
from pyscf.mcpdft.otpd import get_ontop_pair_density
from mrh.my_pyscf.dmet.basistransformation import BasisTransform

'''
In case of MC-PDFT, the idea is to project the dm1 and dm2 calculated
for the embedded space to full molecule/cell ao orbitals. This has to be
done carefully, where one has to take care of the environment and embedded 
space orbitals correctly. One way is to rewrite entire MC-PDFT code to use 
the DMET-PDFT or it can be done like this ! 

We have to change the _dms.casdm1s_to_dm1s function.
'''

'''
Point to note: DMET-PDFT hasn't been tested for 
1. MultiState PDFT
2. Hybrid functionals
3. DMRG Solvers
'''

get_basis_transform = BasisTransform._get_basis_transformed

def casdm1s_to_dm1s(mc, casdm1s, mo_coeff=None, ncore=None, ncas=None):
    '''Convert CAS density matrix to full density matrix'''
    if ncore is None: ncore = mc.ncore
    if ncas is None: ncas = mc.ncas
    if mo_coeff is None: mo_coeff = mc.mo_coeff # These are tagged mo_coeffs from the DMET calculation
    mol = mc.mol # This is for the whole system.

    if not hasattr(mo_coeff, 'ao2lo'):
        raise ValueError('The provided mo_coeff must be tagged with ao2lo')

    dm = mo_coeff.dm
    s = mo_coeff.ovlp
    nelecas = np.trace(casdm1s[0]) + np.trace(casdm1s[1])
    imp_elec = mo_coeff.imp_elec # Total number of embedded electrons
    core_elec = mo_coeff.core_elec # Total number of core electrons
    encore = round((imp_elec - nelecas)/2) # Core orbitals in the embedded space
    ao2co = mo_coeff.ao2co  #
    ao2eo = mo_coeff.ao2eo  #
    embmo = mo_coeff.emb_mo_coeff # neo * neo
    neo = ao2eo.shape[1] # Total number of embedded AOs

    ## Constructing the Core DM for the envioronment then projecting to entire ao space
    if dm.ndim > 2:
        dm = dm[0] + dm[1]
    
    cor2ao = ao2co.T @ s
    core_dm = get_basis_transform(dm, cor2ao.T)
    core_dm = 0.5 * (core_dm + core_dm.T)

    globalrdm = scipy.linalg.block_diag(np.zeros([neo, neo]), core_dm)
    
    # Convert the globalrdm to the AO basis
    ao2lo = np.hstack([ao2eo, ao2co])
    globalrdm1 = get_basis_transform(globalrdm, ao2lo.T)
    globalrdm1 = 0.5 * (globalrdm1 + globalrdm1.T)
    assert np.allclose(np.trace(globalrdm1 @ s), core_elec, atol=1e-5), "The core density matrix is not correct"
   
    # Constructing the core DM for the embedded space then projecting to entire ao space
    u_temp = embmo[:, :encore] # neo * encore
    embcoredm = u_temp @ u_temp.T # neo * neo
    assert np.allclose(np.trace(embcoredm), encore, atol=1e-5), "The core density matrix is not correct"
    # We are spin-separated density matrices, so we need to take care of that
    # by multiplying the core density matrix by 0.5
    globalrdm1 *= 0.5
    globalrdm1  += reduce(np.dot, (ao2eo, embcoredm, ao2eo.T)) # nao * nao
    # Put a condition statement here.

    # Constructing the casdm1s for the embedded space then projecting to entire ao space
    casdm1s = np.asarray(casdm1s) 
    assert casdm1s.ndim > 2

    u_temp = embmo[:, encore:encore+ncas] # neo * ncas
    dm1s_cas_  = [reduce(np.dot, (u_temp, dm_, u_temp.T)) for dm_ in casdm1s] # neo * neo
    dm1s_cas  = [reduce(np.dot, (ao2eo, dm_, ao2eo.T)) for dm_ in dm1s_cas_ ] # nao * nao
    assert np.allclose(np.trace(dm1s_cas[0] @ s) + np.trace(dm1s_cas[1] @ s), nelecas, atol=1e-5), "The CAS density matrix is not correct"
    
    # Assembling the entire spin-separated density matrix
    dm1s = dm1s_cas + globalrdm1[None,:,:]
    assert np.allclose(np.trace(dm1s[0]@s) + np.trace(dm1s[1]@s), sum(mol.nelec), atol=1e-5),"The total density matrix is not correct"

    # # Tags for speeding up rho generators and DF fns
    # no_coeff = mo_coeff[:,:ncore+ncas]
    # no_coeff = np.stack ([no_coeff, no_coeff], axis=0)
    # no_occ = np.zeros ((2,ncore+ncas), dtype=no_coeff.dtype)
    # no_occ[:,:ncore] = 1.0
    # no_cas = no_coeff[:,:,ncore:]
    # for i in range (2):
    #     no_occ[i,ncore:], umat = linalg.eigh (-casdm1s[i])
    #     no_cas[i,:,:] = np.dot (no_cas[i,:,:], umat)
    # no_occ[:,ncore:] *= -1
    # dm1s = lib.tag_array (dm1s, mo_coeff=no_coeff, mo_occ=no_occ)
    return dm1s

def energy_ot (ot, casdm1s, casdm2, mo_coeff, ncore, max_memory=2000, hermi=1):
    energy_ot.__doc__ = mcpdft.otfnal.energy_ot.__doc__

    E_ot = 0.0
    ni, xctype = ot._numint, ot.xctype
    
    if xctype=='HF': return E_ot
    dens_deriv = ot.dens_deriv
    Pi_deriv = ot.Pi_deriv

    nao = mo_coeff.shape[0]
    ncas = casdm2.shape[0]
    cascm2 = _dms.dm2_cumulant (casdm2, casdm1s)
    dm1s = _dms.casdm1s_to_dm1s (ot, casdm1s, mo_coeff=mo_coeff, ncore=ncore,
                                 ncas=ncas)
    # Just because of this.
    # mo_cas = mo_coeff[:,ncore:][:,:ncas]
    if not hasattr(mo_coeff, 'ao2lo'):
        raise ValueError('The provided mo_coeff must be tagged with ao2lo')
    ao2eo = mo_coeff.ao2eo
    embmo = mo_coeff.emb_mo_coeff
    u_temp = embmo[:, ncore:ncore+ncas] # neo * ncas
    mo_cas = ao2eo @ u_temp

    t0 = (logger.process_clock (), logger.perf_counter ())
    make_rho = tuple (ni._gen_rho_evaluator (ot.mol, dm1s[i,:,:], hermi=hermi, with_lapl=False) for
        i in range(2))
    for ao, mask, weight, _ in ni.block_loop (ot.mol, ot.grids, nao,
            dens_deriv, max_memory):
        rho = np.asarray ([m[0] (0, ao, mask, xctype) for m in make_rho])
        t0 = logger.timer (ot, 'untransformed density', *t0)
        Pi = get_ontop_pair_density (ot, rho, ao, cascm2, mo_cas,
            Pi_deriv, mask)
        t0 = logger.timer (ot, 'on-top pair density calculation', *t0)
        if rho.ndim == 2:
            rho = np.expand_dims (rho, 1)
            Pi = np.expand_dims (Pi, 0)
        E_ot += ot.eval_ot (rho, Pi, dderiv=0, weights=weight)[0].dot (weight)
        t0 = logger.timer (ot, 'on-top energy calculation', *t0)

    return E_ot

class _dmet_pdfthelper:
    '''
    Helper class for DMET-PDFT
    '''

    def __init__(self, mc, trans_coeff, mf, **kwargs):
        '''
        This class is written to help the DMET-PDFT calculation.
        Args:
        mc: DMET-MC object
            CASSCF/SA-CASSCF ...
        trans_coeff: dict
            Transformation coefficients for the DMET calculation
        mf: RHF/ROHF object
            mean-field object for the full system
        kwargs: dict
            Additional arguments for flexibility (not used currently)
        '''
        self.mc = mc
        self.trans_coeff = trans_coeff
        self.mf = mf
        self.mol = mf.mol
    
    def _assemble_mo_coeffs(self):
        '''
        Assemble the mo_coeff
        Args:
        Look next to each line for the description of the arguments
        Returns:
        asm_mo: tag_array
            Assembled molecular orbital coefficients for the full system.
        '''

        # Extract the transformation coefficients
        trans_coeff = self.trans_coeff
        mc = self.mc
        mf = self.mf
        ao2lo = trans_coeff['ao2lo']
        ao2co = trans_coeff['ao2co']
        ao2eo = trans_coeff['ao2eo']
        imp_elec = trans_coeff['imp_nelec']
        core_elec = trans_coeff['core_nelec']

        asm_mo = lib.tag_array(np.hstack([ao2eo, ao2co]), 
                                ao2lo=ao2lo, 
                                ao2eo=ao2eo, 
                                ao2co=ao2co, 
                                emb_mo_coeff=mc.mo_coeff,
                                imp_elec=imp_elec, 
                                core_elec=core_elec,
                                dm=mf.make_rdm1(),
                                ovlp=mf.get_ovlp())
        return asm_mo

    def _new_mc(self):
        '''
        Define a new MC object and update the class attributes
        '''
        mc = self.mc
        mf = self.mf

        ncas = mc.ncas
        nelecas = mc.nelecas
        ncore = mf.mol.nelectron - sum(nelecas)

        assert ncore%2 == 0, "The number of core electrons must be even"

        newmc = mcscf.CASCI(mf, ncas, nelecas, ncore=ncore)
        newmc.fcisolver = mc.fcisolver
        newmc.verbose = mc.verbose
        newmc.ci = mc.ci
        newmc.fcisolver.make_rdm1s = lambda *args: mc.fcisolver.make_rdm1s(*args)
        newmc.fcisolver.make_rdm12 = lambda *args: mc.fcisolver.make_rdm12(*args)
        newmc.energy_nuc = lambda* arg: mf.energy_nuc()
        newmc.mo_coeff = self._assemble_mo_coeffs() # Assemble the mo_coeff
        return newmc

    def get_mc(self):
        """
            Return the new MC object
            I am updating the function to make the DMET-PDFT work with current PySCF-Forge.
            Matt, any suggestion to do it more efficently.?
        """
        original_dm1s = mcpdft._dms.casdm1s_to_dm1s
        original_energy_ot = mcpdft.otfnal.energy_ot

        try:
            mcpdft._dms.casdm1s_to_dm1s = casdm1s_to_dm1s
            mcpdft.otfnal.energy_ot = energy_ot
            mcpdft.otfnal.otfnal.energy_ot = energy_ot
            newmc = self._new_mc()
        finally:
            mcpdft._dms.casdm1s_to_dm1s = original_dm1s
            mcpdft.otfnal.energy_ot = original_energy_ot
            mcpdft.otfnal.otfnal.energy_ot = original_energy_ot
        return newmc

dmetpdft = _dmet_pdfthelper

def get_mc_for_dmet_pdft(mc, trans_coeff, mf):
    '''
    Wrapper function
    '''
    mcdecor = dmetpdft(mc, trans_coeff, mf)
    newmc = mcdecor.get_mc()
    return newmc

