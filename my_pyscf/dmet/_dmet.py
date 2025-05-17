import numpy as np
import scipy
from pyscf import lo
from pyscf import gto, ao2mo, lib, scf
from functools import reduce
from mrh.my_pyscf.dmet.localization import Localization
from mrh.my_pyscf.dmet.fragmentation import Fragmentation
from mrh.my_pyscf.dmet.basistransformation import BasisTransform

# Author: Bhavnesh Jangid <jangidbhavnesh@uchicago.edu>

# This class will give you the mean-field object, which can be used to run the MR and post-HF
# calculations.

get_basis_transform = BasisTransform._get_basis_transformed

def is_close_to_integer(num, tolerance=1e-6):
    warning_msg = 'SVD for this RDM has some problem'
    assert (np.abs(num - np.round(num)) < tolerance), warning_msg

class _DMET:
    '''
    Density Matrix Embedding Theory
    '''
    def __init__(self, mf, lo_method='meta_lowdin', bath_tol=1e-6, atmlst=None, atmlabel=None, density_fit=True, **kwargs):
        '''
        Args:
            mf : SCF object
                SCF object for the molecule
            lo : str
                Localization method
            bath_tol : float
                Bath tolerance
            atm_lst : list
                List of atom indices
            atm_label : list
                List of atom labels
            verbose : int
                Print level
        '''
        self.mol = mf.mol
        self.mf = mf
        self.lo_method = lo_method
        self.atmlst = atmlst
        self.atmlabel = atmlabel
        self.bath_tol = bath_tol
        self.loc_rdm1 = None
        self.mask_frag = None
        self.mask_env = None
        self.ao2lo = None
        self.ao2eo = None
        self.ao2co = None
        self.lo2eo = None
        self.lo2co = None
        self.imp_nelec = None
        self.core_nelec = None
        self.density_fit = density_fit
        
    def do_localization(self, **kwargs):
        '''
        Localize the orbitals using the specified method
        '''
        mf = self.mf
        lo_method = self.lo_method
        loc = Localization(mf, lo_method=lo_method)
        ao2lo = loc.get_localized_orbitals()
        loc_rdm1 = loc.localized_rdm1(ao2lo)
        self.ao2lo = ao2lo
        self.loc_rdm1 = loc_rdm1
        return self
    
    def do_fragmentation(self, **kwargs):
        '''
        Fragment the molecule
        '''
        mf = self.mf
        lo_method = self.lo_method
        atmlst = self.atmlst
        atmlabel = self.atmlabel

        frag = Fragmentation(mf, atmlst=atmlst, atmlabel=atmlabel)
        mask_frag, mask_env = frag.get_fragments(atmlst=atmlst, atmlabel=atmlabel)
        self.mask_frag = mask_frag
        self.mask_env = mask_env
        return self 

    def _get_fragment_basis(self):
        '''
        Get the fragment basis
        '''
        frag_basis = self.ao2lo[:, self.mask_frag]
        return frag_basis
    
    def _get_environment_basis(self):
        '''
        Get the environment basis
        '''
        env_basis = self.ao2lo[:, self.mask_env]
        return env_basis
    
    def _construct_bath_and_core(self):
        '''
        Construct the bath and core orbitals.
        '''
        bath_tol = self.bath_tol
        frag_basis = self._get_fragment_basis()
        nfragorb = frag_basis.shape[1]
        nlo = self.ao2lo.shape[1]

        if self.loc_rdm1.ndim > 2:
            # Why it's working with only alpha?
            # Should I take the SVD of the SDM instead of the RDM?
            nelec = self.mol.nelec
            assert len(nelec) == 2, "Electron should have been stored as a tuple (alpha, beta)"
            pos = np.argmax(self.mol.nelec)
            loc_rdm1 = self.loc_rdm1[pos]
        else:
            loc_rdm1 = self.loc_rdm1
        
        env_frag_rdm1 = loc_rdm1[self.mask_env][:, self.mask_frag]
        u, b, vh = np.linalg.svd(env_frag_rdm1, full_matrices=True)
        
        # Selecting the bath orbitals and define the embedding and core space.
        idx_emb = np.where(b > bath_tol)[0]
        idx_core = np.ones(u.shape[1], dtype=bool)
        idx_core[idx_emb] = False 

        # Number of bath and core orbitals
        nbath = len(idx_emb)
        neo = nbath + nfragorb
        ncore = nlo-neo

        # Arrange the rotation matrix for the bath and core orbitals
        u_selected = u[:, idx_emb].copy()
        
        if np.any(idx_core):
            u_core = u[:, idx_core].copy()
        else:
            u_core = np.zeros([np.sum(self.mask_env), ncore], dtype=frag_basis.dtype) 

        return u_selected, u_core
    
    def get_impurity_subspace(self):
        '''
        Get the impurity subspace orbitals
        '''
        
        # Get the mask and rotations matrices
        frag_basis = self._get_fragment_basis()
        env_basis = self._get_environment_basis()
        u_selected, u_core = self._construct_bath_and_core()
        
        # Number of orbitals
        nlo, nfragorb = frag_basis.shape
        nbath = u_selected.shape[1]
        ncore = nlo-nfragorb-nbath

        # Define the impurity subspace
        imp_basis = np.zeros([nlo, nfragorb+nbath], dtype=frag_basis.dtype)
        imp_basis[self.mask_frag, :nfragorb] = np.eye(nfragorb)
        imp_basis[self.mask_env, nfragorb:] = u_selected
        
        # Define the core subspace
        core_basis = np.zeros([nlo, ncore], dtype=frag_basis.dtype)
        core_basis[self.mask_env, :] = u_core

        # Convert localized orbitals to the impurity subspace
        self.lo2eo = imp_basis
        self.lo2co = core_basis
        self.ao2eo = self.ao2lo @ imp_basis
        self.ao2co = self.ao2lo @ core_basis
        return self
    
    def get_imp_nelecs(self):
        '''
        Count the number of electrons in the impurity subspace
        '''
        s = self.mf.get_ovlp()
        dm = self.mf.make_rdm1()
        eo2ao = self.ao2eo.T @ s

        if dm.ndim > 2:
            dm_lo = np.array([get_basis_transform(dm_, eo2ao.T) for dm_ in dm])
            nelecs_spin = [np.trace(dm_) for dm_ in dm_lo]
            for x in nelecs_spin: is_close_to_integer(x)
            nelecs = np.sum(nelecs_spin)

            # Sanity check for the spin
            nalpha, nbeta = self.mf.mol.nelec
            if nalpha != nbeta:
                assert np.isclose(abs(nelecs_spin[1] - nelecs_spin[0]), abs(nalpha - nbeta), atol=1e-6),\
                                "Impurity subspace should have the same spin as the molecule. \
                                Non-zeros spin for the environment is not implemented yet."
        else:
            dm_lo = get_basis_transform(dm, eo2ao.T)
            nelecs = np.trace(dm_lo)
            is_close_to_integer(nelecs)

        # Set up this value.
        self.imp_nelec = int(round(nelecs))
    
        return self

    def get_core_elecs(self):
        '''
        Get the core electrons
        '''
        s = self.mf.get_ovlp()
        dm = self.mf.make_rdm1()
        cor2ao = self.ao2co.T @ s

        if dm.ndim > 2:
            ncore = np.sum([np.trace(get_basis_transform(dm_, cor2ao.T)) for dm_ in dm])
        else:
            ncore = np.trace(get_basis_transform(dm, cor2ao.T))

        self.core_nelec = int(round(ncore))

        return self

    def dump_flags(self):
        '''
        Print the flags
        '''
        log = lib.logger.new_logger(self, self.mf.verbose)
        log.info('')
        log.info('************************************************')
        log.info('******* Density Matrix Embedding Theory ********')
        log.info('************************************************')
        log.info("******** System's Information ********")
        log.info('Number of cGTOs = %s', self.mol.nao)
        log.info('Number of elecs = %s', self.mol.nelectron)
        log.info('Number of atoms = %s', self.mol.natm)
        log.info("******** Fragment's Information ********")
        log.info('Fragment type = %s', 'atom list' if self.atmlst is not None else 'atom label')
        log.info('Lo_method = %s', self.lo_method)
        log.info('Bath_tol = %s', self.bath_tol)
        log.info('Number of frag orb = %s', sum(self.mask_frag))
        log.info('Number of env orb = %s',  sum(self.mask_env))
        log.info('Number of imp. electrons = %s',  round(self.imp_nelec))
        log.info('Number of core electrons = %s',  round(self.core_nelec))
        log.info('Number of imp orb = %s',  self.ao2eo.shape[1])
        log.info('Number of bath orb = %s', self.ao2eo.shape[1] - sum(self.mask_frag))
        log.info('Number of core orb = %s', self.ao2co.shape[1])
        return self

    def _dummy_mol(self):
        '''
        Create a dummy molecule
        '''
        mol = gto.M()
        mol.atom = 'He 0 0 0'
        mol.spin = self.mol.spin
        mol.verbose = self.mf.verbose
        mol.max_memory = self.mol.max_memory
        mol.output = self.mol.output
        return mol
    
    def get_veff(self, eri, dm):
        '''
        Get the J and K terms.
        Currently, I am using the RHF object by setting the eri
        to get the J and K terms.
        '''
        mf = self.mf

        if hasattr(mf, 'with_df') and mf.with_df is not None and self.density_fit:
            mftemp = scf.ROHF(self.mol).density_fit()
            mftemp.with_df._cderi = eri
            j, k = mftemp.with_df.get_jk(dm=dm, hermi=1)
            del mftemp
        else:
            mftemp = scf.ROHF(self.mol)
            mftemp._eri = eri
            j, k = mftemp.get_jk(dm=dm, hermi=1)
            del mftemp

        if dm.ndim > 2:
            j = j[0] + j[1]
            veff = np.asarray([j-k[0], j-k[1]])
        else:
            veff = np.asarray(j - 0.5 * k)

        assert veff.shape == dm.shape, "Shape of veff and dm should be same."
        
        return veff

    def _get_dmet_mf(self):
        '''
        Get the DMET mean-field object
        '''
        mf = self.mf
        neo = self.ao2eo.shape[1]
        s = mf.get_ovlp()
        ao2eo = self.ao2eo
        ao2co = self.ao2co
        eo2ao = self.ao2eo.T @  s

        basistransf = BasisTransform(mf, ao2eo, ao2co)

        if hasattr(mf, 'with_df') and mf.with_df is not None and self.density_fit:
            eri = basistransf._get_cderi_transformed(ao2eo)
        else:
            eri = ao2mo.restore(8, basistransf._get_eri_transformed(ao2eo=ao2eo), neo)
        
        fock = basistransf._get_fock_transformed()
       
        dm = mf.make_rdm1()

        if fock.ndim > 2:
            dm_guess = np.asarray([get_basis_transform(dm_, eo2ao.T) for dm_ in dm])
            nelecs = sum([np.trace(dm_) for dm_ in dm_guess])
            veff = self.get_veff(eri, dm_guess)
            fock -= veff
            fock  = 0.5 * (fock[0] + fock[1])
        else:
            dm_guess = get_basis_transform(dm, eo2ao.T)
            veff = self.get_veff(eri, dm_guess)
            nelecs = np.trace(dm_guess)
            fock -= veff
        
        emb_mol = self._dummy_mol()
        emb_mol.nelectron = round(nelecs)
        emb_mol.max_memory = mf.mol.max_memory
        emb_mol.build()

        # Core energy contribution
        core_energy = self._get_core_contribution(ao2eo=ao2eo, ao2co=ao2co)

        if hasattr(mf, 'with_df') and mf.with_df is not None and self.density_fit:
            emb_mf = scf.ROHF(emb_mol).density_fit()
            emb_mf.with_df._cderi = eri
        else:
            emb_mf = scf.ROHF(emb_mol)
            emb_mf._eri = eri

        emb_mf.get_hcore = lambda *args: fock
        emb_mf.get_ovlp  = lambda *args: np.eye(neo)
        emb_mf.conv_tol = 1e-10
        emb_mf.max_cycle = 100
        emb_mf.energy_nuc = lambda *args: core_energy
        emb_mf.kernel(dm_guess)

        assert emb_mf.converged, 'DMET mean-field did not converge'

        return emb_mf

    def runDMET(self):
        '''
        Kernel to run the DMET calculations

        1. Localize the orbitals
        2. Fragment the molecule
        3. Get the impurity subspace
        4. Sanity checks for the number of electrons
        5. Run the DMET mean-field
        
        Returns:
            dmet_mf : SCF object
                DMET mean-field object
            ao2eo : np.array
                Transformation matrix from AO to EO
            ao2co : np.array
                Core coefficient matrix
        '''
        self.do_localization()
        self.do_fragmentation()
        self.get_impurity_subspace()
        self.get_imp_nelecs()
        self.get_core_elecs()
        self.dump_flags()
        dmet_mf = self._get_dmet_mf()
        return dmet_mf
    
    def _get_core_contribution(self, ao2eo=None, ao2co=None):
        '''
        Calculate the core energy
        CoreEnergy = Nuclear Repulsion Energy + Energy from the core orbitals at the mean-field level.
        '''
        
        mf = self.mf
        neo = self.ao2eo.shape[-1]
        s = mf.get_ovlp()
        
        if ao2eo is None:
            ao2eo = self.ao2eo
        if ao2co is None:
            ao2co = self.ao2co
        
        '''
        Core orbitals are doubly occupied therefore, can get 
        away with this.
        '''
        dm_full_ao = dm = mf.make_rdm1()
        if dm.ndim > 2:
            dm = dm[0] + dm[1]
       
        cor2ao = ao2co.T @ s
        core_dm = get_basis_transform(dm, cor2ao.T)
       
        core_dm = 0.5 * (core_dm + core_dm.T)

        globalrdm = scipy.linalg.block_diag(np.zeros([neo, neo]), core_dm)
       
        # Convert the globalrdm to the AO basis
        ao2eo = np.hstack([ao2eo, ao2co])
        globalrdm1 = get_basis_transform(globalrdm, ao2eo.T)
        globalrdm1 = 0.5 * (globalrdm1 + globalrdm1.T)

        # This piece of code can be further optimized.
        h1e = mf.get_hcore()
       
        if dm_full_ao.ndim > 2:
            veff = mf.get_veff(dm=globalrdm1)
            h1e += 0.25 * (veff[0] + veff[1])
            del veff
        else:
            h1e += 0.5 * mf.get_veff(dm=globalrdm1)

        energy  = np.einsum('ij, ij->', h1e, globalrdm1) 
        energy += mf.energy_nuc()

        del core_dm, globalrdm, globalrdm1, h1e, cor2ao, ao2eo

        return energy

    def assemble_mo(self, mc_mo_coeff):
        '''
        Assemble the mo_coeff to run the PDFT with the dmet_mf object.
        args:
            mf: RHF/ROHF object
                mean-field object for the full system
            ao2eo: np.array (nao, neo)
                transformation matrix from the full system to the embedded system. Note that
                nao: number of orbitals in the full system
                neo: number of orbitals in the embedded system
                ncore: number of core orbitals from the environment. (Don't get confuse with the ncore of mcscf)
                nao = neo + ncore
            ao2co: np.array (nao, ncore)
                transformation matrix from the full system to the core space
            mc_mo_coeff: np.array (neo, neo)
                mo_coeff for the embedded CASSCF calculation
        returns:
            mo_coeff: np.ndarray
                mo_coeff for the full system
        '''
        mf = self.mf
        ao2co = self.ao2co
        ao2eo = self.ao2eo
        
        dm = mf.make_rdm1()
        s = mf.get_ovlp()
        
        cor2ao = ao2co.T @ s

        if dm.ndim > 2:
            dm = dm[0] + dm[1]
        
        # Generate the core density matrix and using that transform the ao2co
        # to the canonical basis.
        core_dm = get_basis_transform(dm, cor2ao.T)
        e, eigvec = np.linalg.eigh(core_dm)
        sorted_indices = np.argsort(e)[::-1]
        eigvec_sorted = eigvec[:, sorted_indices]  
        ao2co = ao2co @ eigvec_sorted
        core_nelec = int(round(np.sum(e)))
        assert core_nelec % 2 == 0, "Core nelec should be even., Something went wrong."
        ao2eo = ao2eo @ mc_mo_coeff

        ncore = core_nelec//2
        neo = ao2eo.shape[1]

        # Now we can assemble the full space mo_coeffs.
        mo_coeff = np.empty_like(mf.mo_coeff)
        mo_coeff[:, :ncore] = ao2co[:, :ncore]
        mo_coeff[:, ncore:ncore+neo] = ao2eo
        mo_coeff[:, ncore+neo:] = ao2co[:, ncore:]
        return mo_coeff
    
    # For pyscf like interface
    def kernel(self):
        dmet_mf = self.runDMET()
        return dmet_mf