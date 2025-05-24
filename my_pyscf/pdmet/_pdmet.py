import numpy as np
from pyscf import ao2mo
from pyscf.pbc import gto, scf
from functools import reduce
from mrh.my_pyscf.pdmet import basistransformation as bt
from mrh.my_pyscf.pdmet.localization import Localization
from mrh.my_pyscf.pdmet.fragmentation import Fragmentation
from mrh.my_pyscf.dmet._dmet import _DMET as molDMET
# Author: Bhavnesh Jangid <jangidbhavnesh@uchicago.edu>

get_basis_transform = bt.BasisTransform._get_basis_transformed

class _pDMET(molDMET):
    '''
    DMET class for PBC Supercell:
    Child Class of Molecular DMET class.
    '''
    def __init__(self, kmf, lo_method='meta_lowdin', bath_tol=1e-6, atmlst=None, density_fit=False, **kwargs):
        super().__init__(kmf, lo_method=lo_method, bath_tol=bath_tol, atmlst=atmlst, density_fit=density_fit, **kwargs)
        self.cell = kmf.cell
        
    def do_localization_(self, **kwargs):
        '''
        Localize the entire orbital space. Then assigning
        the rdm1 and mo_coeff to instance.
        '''
        loc = Localization(self.mf, lo_method=self.lo_method)
        ao2lo = loc.get_localized_orbitals()
        loc_rdm1 = loc.localized_rdm1(ao2lo)
        # Now set these values.
        self.ao2lo = ao2lo
        self.loc_rdm1 = loc_rdm1
        return self
    
    def do_fragmentation_(self, **kwargs):
        '''
        Fragment the molecule
        '''
        frag = Fragmentation(self.mf, atmlst=self.atmlst)
        mask_frag, mask_env = frag.get_fragments(atmlst=self.atmlst)
        # Assign the fragment and env orbitals.
        self.mask_frag = mask_frag
        self.mask_env = mask_env
        return self 
    
    def _dummy_cell(self):
        '''
        Create a dummy cell
        '''
        cell = gto.Cell()
        cell.atom = self.cell.atom
        cell.a = self.cell.a
        cell.spin = self.cell.spin
        cell.verbose = self.cell.verbose
        cell.max_memory = self.cell.max_memory
        cell.output = self.cell.output
        return cell
    
    def get_veff(self, cell, eri, dm, density_fit=False):
        '''
        Get the veff terms
        '''
        mftemp = scf.RHF(cell, exxdiv = None).density_fit()
        if density_fit:
            mftemp.with_df._cderi = eri
        else:
            mftemp._eri = eri
        veff = mftemp.get_veff(dm=dm, hermi=1)
        del mftemp
        return veff

    def _get_dmet_mf(self):
        '''
        Get the DMET mean-field object
        '''
        kmf = self.mf
        neo = self.ao2eo.shape[1]
        ao2eo = self.ao2eo
        ao2co = self.ao2co
        density_fit = self.density_fit

        s = kmf.get_ovlp()
        eo2ao = ao2eo.T @ s

        basistransf = bt.BasisTransform(kmf, ao2eo, ao2co)

        if density_fit:
            eri = basistransf._get_cderi_transformed(ao2eo)
        else:
            eri = ao2mo.restore(1, basistransf._get_eri_transformed(ao2eo=ao2eo), neo)
            eri =  basistransf._get_eri_transformed(ao2eo=ao2eo)

        fock = basistransf._get_fock_transformed()
       
        dm_full_ao = kmf.make_rdm1()
    
        if fock.ndim > 2:
            dm_guess = np.asarray([get_basis_transform(dm, eo2ao.T) for dm in dm_full_ao])
            nelecs = sum([np.trace(dm_) for dm_ in dm_guess])
        else:
            dm_guess = get_basis_transform(dm_full_ao, eo2ao.T)
            nelecs = np.trace(dm_guess)
        
        emb_cell = self._dummy_cell()
        emb_cell.nelectron = round(nelecs)
        emb_cell.max_memory = kmf.cell.max_memory
        emb_cell.nao_nr = lambda *arg: neo
        emb_cell.build()

        if fock.ndim > 2:
            veff = self.get_veff(emb_cell, eri, dm_guess, density_fit=density_fit)
            fock -= veff
            fock  = 0.5 * (fock[0] + fock[1])
        else:
            dm_guess = get_basis_transform(dm_full_ao, eo2ao.T)
            veff = self.get_veff(emb_cell, eri, dm_guess,  density_fit=density_fit)
            fock -= veff

        # Contribution of the core energy to the total energy
        core_energy = self._get_core_contribution(ao2eo=ao2eo, ao2co=ao2co)

        emb_mf = scf.ROHF(emb_cell).density_fit()
        
        if density_fit:
            '''
            Basically, the eri will be created from the 3-index DF integrals and 
            will be used to solve the embedding problem using any high-level method.
            In case of CAS, even if we use the density fitting, CAS algorithm will 
            create the 4-index eri containing upto 2 orbitals belongs to the embedding space. 
            '''
            emb_mf.with_df._cderi = eri
        else:
            emb_mf._eri = eri 

        emb_mf.exxdiv = None
        emb_mf.get_hcore = lambda *args: fock
        emb_mf.get_ovlp  = lambda *args: np.eye(neo)
        emb_mf.conv_tol = 1e-10
        emb_mf.max_cycle = 100
        emb_mf.energy_nuc = lambda *args: core_energy
        emb_mf.kernel(dm_guess)

        if not emb_mf.converged:
            emb_mf = emb_mf.newton()
            emb_mf.kernel()

        assert emb_mf.converged, 'DMET mean-field did not converge'
        return emb_mf
    
    def _get_core_contribution(self, ao2eo=None, ao2co=None):
        '''
        Construct the core energy
        '''
        import scipy
        neo = self.ao2eo.shape[-1]

        if ao2eo is None:
            ao2eo = self.ao2eo
        if ao2co is None:
            ao2co = self.ao2co

        '''
        Core orbitals are doubly occupied therefore, can get 
        away with this.
        '''
        dm_full_ao = self.mf.make_rdm1()
        if dm_full_ao.ndim > 2:
            dm = dm_full_ao[0] + dm_full_ao[1]
        else:
            dm = dm_full_ao

        cor2ao = ao2co.T @ self.mf.get_ovlp()

        core_dm = reduce(np.dot, (cor2ao, dm, cor2ao.T))
        core_dm = 0.5*(core_dm + core_dm.T)
        globalrdm = scipy.linalg.block_diag(np.zeros([neo, neo]), core_dm)
        
        ao2eo = np.hstack([ao2eo, self.ao2co])
        
        globalrdm1 = reduce(np.dot, (ao2eo, globalrdm, ao2eo.T))

        # This piece of code can be further optimized.
        h1e = self.mf.get_hcore()
       
        if dm_full_ao.ndim > 2:
            h1e += 0.25 * (self.mf.get_veff(dm=globalrdm1)[0] + self.mf.get_veff(dm=globalrdm1)[1])
        else:
            h1e += 0.5 * self.mf.get_veff(dm=globalrdm1)
        energy  = np.einsum('ij, ij->', h1e, globalrdm1) 
        energy += self.mf.energy_nuc()
        return energy
    
    # For pyscf like interface
    def kernel(self):
        dmet_mf = self.runDMET()
        return dmet_mf
