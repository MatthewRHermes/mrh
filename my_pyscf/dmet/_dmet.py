import numpy as np
import scipy
from pyscf import gto, ao2mo, lib, scf
from mrh.my_pyscf.dmet.localization import Localization
from mrh.my_pyscf.dmet.fragmentation import Fragmentation
from mrh.my_pyscf.dmet.basistransformation import BasisTransform

# Author: Bhavnesh Jangid <jangidbhavnesh@uchicago.edu>

# This class will give you the mean-field object, which can be used to run the MR and post-HF
# calculations.

get_basis_transform = BasisTransform._get_basis_transformed

def is_close_to_integer(num, tolerance=1e-6):
    warning_msg = f'SVD for this RDM has some problem {(np.abs(num - np.round(num)) < tolerance)}'
    assert (np.abs(num - np.round(num)) < tolerance), warning_msg 


def perform_schmidt_decomposition_type1(rdm, nfragorb, nlo, bath_tol=1e-5):
    '''
    This way of schmidt decomposition corresponds to previous implementation
    where one will consider all the singly occupied orbital in the bath, whether they
    are in impurity space or not.

    Not a wise way of doing it, still having it to make sure we have functionalies of previous
    implementations.

    Args:
        rdm: np.ndarray (2, n, n); 
             n=number of enviroment orbitals
        nfragorb: int
            no of fragment orb
        nlo: int
            no of localized orb, basically this will be same as that of total number of atomic orbital.
        bath_tol: float
            threshold to keep an orbital in the embedding space or through it out.
    return
        u_selected: np.array (nlo, neo)
            unitary matrix which will transform the localized orbitals to the bath orbitals 
        u_core: np.array(nlo, ncore)
            unitary matrix which will transform the localized orbitals to the core orbitals 
    '''
    alpha_evecs, alpha_evals = np.linalg.svd(rdm[0])[:2]
    beta_evecs, beta_evals= np.linalg.svd(rdm[1])[:2]
    occ, u, alpha = (alpha_evals, alpha_evecs, 1) \
        if np.sum(alpha_evals) >= np.sum(beta_evals) else (beta_evals, beta_evecs, 0)

    # Select all the orbitals which are above the bath_tolerance
    # in this way, if their are any singly occupied orbital in environment
    # that will taken into the bath orbital.
    sinorbthresh = 0.9
    occ1, occ2 = occ, beta_evals if alpha else alpha_evals
    bathocc = np.array([(ai - bi) if ai > sinorbthresh and bi > sinorbthresh \
        else ai for ai, bi in zip(occ1, occ2)])

    idx_emb = np.where(bathocc > bath_tol)[0]
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
        u_core = np.zeros([nlo-nfragorb, ncore], dtype=rdm.dtype)
    return u_selected, u_core


class _DMET:
    '''
    Density Matrix Embedding Theory
    '''
    def __init__(self, mf, lo_method='meta_lowdin', bath_tol=1e-6, atmlst=None, density_fit=True, **kwargs):
        _keys = ['loc_rdm1','mask_frag','mask_env','ao2lo','ao2eo',
                 'ao2co','lo2eo','lo2co','imp_nelec','core_nelec',]
        '''
        Args:
            mf : SCF object
                SCF object for the molecule
            lo_method : str
                Localization method
            bath_tol : float
                Bath tolerance
            atm_lst : list
                List of atom indices
            density_fit: boolean
                DF option for the embedded part
            verbose : int
                Print level
            nao: int
                number of atomic orb. (cGTOs/basis)
            nlo: int
                number of localized orb.
            neo: int
                number of embedded orb (Impurity + bath)
            loc_rdm1: np.array (nao,nao)
                mean-field 1RDM in localized orb basis
            mask_frag: vector (nao,)
                orbital index which are centered on fragment
            mask_env: vector(nao,)
                orbital index for the environment orbitals
            ao2lo: np.array (nao,nlo)
                localized orbital coeffcient
            ao2eo: np.array (nao,neo)
                embedding orbital coeffcient
            ao2co: np.array (nao,nlo-neo)
                core orbital coeffcient
            lo2eo: np.array (nlo, neo)
                transf. matrix for ao2lo->ao2eo
            lo2eo: np.array (nlo, nlo-neo)
                transf. matrix for ao2lo->ao2co
            imp_nelec: int
                number of electrons in embedding subspace
            core_nelec: int
                number of electrons in core subspace
        '''
        self.mol = mf.mol
        self.mf = mf
        self.lo_method = lo_method
        self.atmlst = atmlst
        self.bath_tol = bath_tol
        self.density_fit = density_fit
       
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

    def _get_fragment_basis(self):
        '''
        Get the fragment basis
        '''
        frag_basis = self.ao2lo[:, self.mask_frag]
        return frag_basis
    
    def _get_environment_basis(self):
        '''
        Get environment orbitals
        '''
        env_basis = self.ao2lo[:, self.mask_env]
        return env_basis
    
    def _check_env_basis_for_single_orb(self, loc_rdm1):
        '''
        If environment has the singly occupied orbital
        return 1, which will direct the program to use the
        previous schmidt decomposition technique.
        '''
        rdm = np.array([loc_rdm1[x][self.mask_env][:, self.mask_env] for x in range(2)])
        alpha_evecs, alpha_evals = np.linalg.svd(rdm[0])[:2]
        beta_evecs, beta_evals = np.linalg.svd(rdm[1])[:2]
        if np.max(np.sort(alpha_evals) - np.sort(beta_evals)) < 1e-4:
            return 0
        else:
            lib.logger.warn(self.mf, 'Your environment has the singly occupied orbital.'
                        'Please choose your fragmentation wisely.')
            return 1

    @staticmethod
    def perform_schmidt_decomposition_type1(rdm, nfragorb, nlo, bath_tol=1e-5):
        return perform_schmidt_decomposition_type1(rdm, nfragorb, nlo, bath_tol=1e-5)

    def perform_schmidt_decomposition(self):
        '''
        Construct the bath and core orbitals.
        '''
        bath_tol = self.bath_tol
        frag_basis = self._get_fragment_basis()
        nfragorb = frag_basis.shape[1]
        nlo = self.ao2lo.shape[1]

        if self.loc_rdm1.ndim > 2:
            # Switch to previous SVD 
            # As in by fiat keep all the singly occupied orbitals in the bath orbital.
            # Only perform this if env has more than 0 orbitals.
            if sum(self.mask_env) and self._check_env_basis_for_single_orb(self.loc_rdm1):
                rdm = np.array([self.loc_rdm1[x][self.mask_env][:, self.mask_env] for x in range(2)])
                u_selected, u_core = perform_schmidt_decomposition_type1(rdm, nfragorb, nlo, bath_tol=self.bath_tol)
                return u_selected, u_core

            nelec = self.mol.nelec
            assert len(nelec) == 2
            pos = np.argmax(self.mol.nelec)
            loc_rdm1 = self.loc_rdm1[pos]
        else:
            loc_rdm1 = self.loc_rdm1
        
        env_frag_rdm1 = loc_rdm1[self.mask_env][:, self.mask_frag]
        u, b = np.linalg.svd(env_frag_rdm1, full_matrices=True)[:2]
        
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
    
    def generate_impurity_subspace_(self):
        '''
        Get the impurity subspace orbitals
        '''
        # Get the mask and rotations matrices
        frag_basis = self._get_fragment_basis()
        u_selected, u_core = self.perform_schmidt_decomposition()
        
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
        log.info('Fragment type = %s', 'atom list' )
        log.info('Lo_method = %s', self.lo_method)
        log.info('Bath_tol = %s', self.bath_tol)
        log.info('Number of frag orb = %s', sum(self.mask_frag))
        log.info('Number of env orb = %s',  sum(self.mask_env))
        log.info('Number of imp orb = %s',  self.ao2eo.shape[1])
        log.info('Number of bath orb = %s', self.ao2eo.shape[1] - sum(self.mask_frag))
        log.info('Number of core orb = %s', self.ao2co.shape[1])
        log.info('Number of imp. electrons = %s',  round(self.imp_nelec))
        log.info('Number of core electrons = %s',  round(self.core_nelec))
        return self

    def _dummy_mol(self):
        '''
        Create a dummy molecule
        '''
        mol = gto.M()
        mol.atom = 'He 0 0 0'
        mol.spin = self.mol.spin
        mol.verbose = self.mf.verbose
        mol.max_memory = self.mol.max_memory # Could assign the remaining memory, but I am lazy.
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
        self.do_localization_()
        self.do_fragmentation_()
        self.generate_impurity_subspace_()
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

        globalrdm = get_basis_transform (core_dm, ao2co.T)
        globalrdm = 0.5 * (globalrdm + globalrdm.T)

        # This piece of code can be further optimized.
        h1e = mf.get_hcore()
       
        if dm_full_ao.ndim > 2:
            veff = mf.get_veff(dm=globalrdm)
            h1e += 0.25 * (veff[0] + veff[1])
            del veff
        else:
            h1e += 0.5 * mf.get_veff(dm=globalrdm)

        energy  = np.einsum('ij, ij->', h1e, globalrdm) 
        energy += mf.energy_nuc()

        del core_dm, globalrdm, h1e, cor2ao, ao2eo

        return energy

    def assemble_mo(self, mc_mo_coeff):
        '''
        Assemble the mo_coeff to run the PDFT with the dmet_mf object.
        args:
            mf: RHF/ROHF object
                mean-field object for the full system
            ao2eo: np.array (nao, neo)
                transformation matrix from the full system to the 
                        embedded system. Note that
                nao: number of orbitals in the full system
                neo: number of orbitals in the embedded system
                ncore: number of core orbitals from the environment.
                         (Don't get confuse with the ncore of mcscf)
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
        mo_coeff = np.zeros_like(mf.mo_coeff)
        mo_coeff[:, :ncore] = ao2co[:, :ncore]
        mo_coeff[:, ncore:ncore+neo] = ao2eo
        mo_coeff[:, ncore+neo:] = ao2co[:, ncore:]
        return mo_coeff
    
    # For pyscf like interface
    def kernel(self):
        dmet_mf = self.runDMET()
        return dmet_mf