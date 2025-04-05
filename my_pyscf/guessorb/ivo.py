import numpy as np
import scipy
from functools import reduce
from pyscf import lib, scf

# Improved Virtual Orbitals (IVOs)
# Ref: J. Chem. Phys. 114, 2592–2600 (2001)


DEG_THRESH = 1e-3

class ivo:
    '''
    For a given set of the virtual orbitals, this will add the IVO contributions.
    They are the eigen-vectors of the virtual fock matrix, constructed from the excitation of
    the electrons from the HOMO or near-HOMO orbitals to LUMO orbitals.
    '''
    def __init__(self, mf, mo_energy, mo_occ, mo_coeff):
        self.mf = mf
        self.mol = mf.mol
        self.mo_energy = mo_energy
        self.mo_occ = mo_occ
        self.mo_coeff = mo_coeff
    
    def _diagonalization(self, fock):
        '''
        Digonalize the fock matrix
        Args:
            fock: fock matrix
        Returns:
            e_sorted: sorted mo_energy
            c_sorted: sorted mo_coeff
        '''
        e, c = scipy.linalg.eigh(fock)
        sorted_indices = np.argsort(e)
        e_sorted = e[sorted_indices]
        c_sorted = c[:, sorted_indices]
        return e_sorted, c_sorted
        
    def _make_rdm1(self, mo_coeff, orbind):
        '''
        Construction of the density matrix for a given orbital index
        Args:
            mo_coeff: molecular orbital coefficients(nao, nmo)
            orbind: orbital index (int)
        Returns:
            dm: density matrix (nao, nao)
        '''
        nmo = mo_coeff.shape[1]
        mo_occ = np.zeros(nmo)
        mo_occ[orbind] = 2
        mocc = mo_coeff[:, orbind].reshape(-1,1)
        dm = 2 * mocc @ mocc.conj().T
        return dm

    def _get_jk(self, mol, dm):
        '''
        Get the J and K matrices for a given density matrix
        Args:
            mol: molecule object
            dm: density matrix (nao, nao)
        Returns:
            j: J matrix
            k: K matrix
        '''
        j, k = scf.hf.get_jk(mol, dm, hermi=1)
        return j, k

    def _get_virtual_fock(self, mf, mo_coeff, mo_occ):
        '''
        Get the virtual fock matrix
        Args:
            mf: mean-field object
            mo_coeff: molecular orbital coefficients(nao, nmo)
            mo_occ: molecular orbital occupation numbers
        Returns:
            fock_vir: virtual fock matrix(nmo, nmo)
        '''
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        fock = mf.get_fock(dm=dm)
        vir_idx  = np.where(mo_occ == 0)[0]

        assert np.count_nonzero(vir_idx) > 0, "No virtual orbitals found"

        orb = mo_coeff[:, vir_idx]
        fock_vir = reduce(np.dot, (orb.T, fock, orb))

        return fock_vir

    def _add_ivo_contributions(self, mf, fock_vir, mo_coeff, mo_occ, mo_energy):
        '''
        F' = F + IVO
        IVO = -J + 2K 
        (For the excitation of electrons from occupied to virtual orbitals)    
        Add the IVO contributions to the fock matrix
        Args:
            mf: mean-field object
            fock_vir: virtual fock matrix(nmo, nmo)
            mo_coeff: molecular orbital coefficients(nao, nmo)
            mo_occ: molecular orbital occupation numbers
            mo_energy: molecular orbital energies
        Returns:
            modified_mo_coeff: modified molecular orbital coefficients
            modified_mo_energy: modified molecular orbital energies
        '''
        mol = self.mol
        modified_mo_coeff = mo_coeff.copy()
        modified_mo_energy = mo_energy.copy()

        e_sort = mo_energy[np.argsort(mo_energy)]
        nocc = mol.nelectron // 2
        homo_e = e_sort[nocc-1]
        
        vir_idx  = np.where(mo_occ == 0)[0]
        orb = mo_coeff[:, vir_idx]

        orbindexes = [i for i in range(nocc) if abs(e_sort[i] - homo_e) < DEG_THRESH]

        fock_ivo = np.zeros_like(fock_vir)

        for orbind in orbindexes:
            dm_o = self._make_rdm1(mo_coeff, orbind)
            j, k = self._get_jk(mf.mol, dm_o)
            f_ivo = 2. * k - j
            fock_ivo += 1/len(orbindexes) * reduce(np.dot, (orb.T, f_ivo, orb))

        fock_vir += fock_ivo

        e, c = self._diagonalization(fock_vir)
        modified_mo_coeff[:, vir_idx] = np.dot(orb, c)
        modified_mo_energy[vir_idx] = e
    
        return modified_mo_coeff, modified_mo_energy

    def kernel(self, mf, mo_energy, mo_coeff, mo_occ):
        '''
        IVO Kernel
        '''
        
        fock_vir = self._get_virtual_fock(mf, mo_coeff, mo_occ)
        mod_mo_coeff, mod_mo_energy = self._add_ivo_contributions(mf, fock_vir,mo_coeff, mo_occ, mo_energy)
        
        return mod_mo_coeff, mod_mo_energy


def get_ivo(mf, mo_energy, mo_coeff, mo_occ):
    '''
    Wrapper function for the IVO
    Args:

    Returns:
        mo:
        mo_e:
    '''
    log = lib.logger.new_logger(lib.StreamObject, mf.verbose)
    if isinstance(mf, scf.rohf.ROHF):
        raise NotImplementedError('ROHF is not supported yet. Will do it later.') 
        
    elif isinstance(mf, scf.hf.RHF):
        ivo_obj = ivo(mf, mo_energy, mo_occ, mo_coeff)
        mo, mo_e = ivo_obj.kernel(mf, mo_energy, mo_coeff, mo_occ)

    elif isinstance(mf, scf.uhf.UHF):
        log.info('UHF mean-field object detected. Converting to RHF/ROHF.')
        mf = mf.to_rhf()
        mo_energy = mf.mo_energy
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        ivo_obj = ivo(mf, mo_energy, mo_occ, mo_coeff)
        mo, mo_e = ivo_obj.kernel(mf, mo_energy, mo_coeff, mo_occ)

    else:
        raise ValueError('Unsupported mean-field object type. Only RHF, ROHF, and UHF are supported.')

    return mo, mo_e
