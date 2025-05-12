import copy
import json
import scipy
import numpy as np
from scipy import linalg
from functools import reduce
from collections import Counter
from pyscf import gto, scf, lib
import mrh
import os

# Some standard cutoffs required for the guessorb
VIR_E_CUTOFF = -1e-3 # In Hartree
REF_BASIS = 'ano-rcc' # Basis
VIR_E_SHIFT = 3.0 # In Hartree

# List of atoms. It should be somewhere in pyscf or mrh then it will
# be straightforward import.
element_symbols = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "I", "Te", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"]

def loadorbitalenergy():
    '''
    Load the orbital energies listed in the basis set file of OpenMolcas
    and extracted in json file.
    '''
    orbitalenergyfile = os.path.join(os.path.dirname(mrh.__file__),\
            'my_pyscf/guessorb/orbitalenergy.json')
    with open(orbitalenergyfile, 'r') as f:
        orbenergy = json.load(f)
    return orbenergy

def get_model_fock_atom(atm1, atm2, symb):
    '''
    Model Fock Matrix in AO Basis.
    args:
        atm1: 
            mol object for an atom
            total basis: nao
        atm2: 
            mol object for same atom but in 
            ref. basis function (ANO-RCC)
        symb: str
            pure atomic sysbol
    return:
        fock: nao*nao
            This will be in atomic orbital basis

    Note that: PySCF ANO-RCC basis has some problem. mol.bas_nctr(l) 
    is giving incorrect no of cgto because they further split the same shell.
    '''

    l_shell_ref = dict(Counter([x[0] for x in atm2._basis[symb]]))
    nctr_per_l_ref = [atm2.bas_nctr(l) + (atm2.bas_nctr(l + 1) \
                        if l_shell_ref[l] > 1 else 0)
                        for i, l in enumerate(l_shell_ref.keys())]

    # Get the overlap matrices
    s1 = atm1.intor_symmetric('int1e_ovlp')
    s1_inv = np.linalg.inv(s1)
    s12 = gto.intor_cross('int1e_ovlp', atm1, atm2)
 
    # Load the orbital energies of given atom in reference basis.
    orbital_energies = loadorbitalenergy()

    ModelFockFinal = []
    for nctr, l in enumerate(l_shell_ref.keys()):
        nctr_l = nctr_per_l_ref[nctr]
        deg = 2*l + 1
        fock_l = np.zeros([deg*nctr_l, deg*nctr_l], dtype=s1.dtype)
        orb_ene = orbital_energies[symb][str(l)]
        orbene_l = np.array(orb_ene * deg)
        np.fill_diagonal(fock_l[:len(orbene_l)], np.sort(orbene_l))
        ModelFockFinal.append(fock_l)

    # Project the Model Fock matrix from the ref basis to
    # original basis. 
    modelFock  = linalg.block_diag(*ModelFockFinal)
    modelFock_ = reduce(np.dot, (s12, modelFock, s12.T))
    modelFock  = reduce(np.dot, (s1_inv, modelFock_, s1_inv))
    
    del ModelFockFinal, modelFock_, orbital_energies
    
    return modelFock

def get_model_fock(mol):
    '''
    Create the Model Fock Matrix for the given mole object.
    args:
        mol:
            mole object
    returns:
        fock_ao: nao x nao ndarray
            model fock matrix in ao basis
    '''

    atoms_fock = []
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        purssymb = mol.atom_pure_symbol(ia)
        atombasis = mol._basis[symb]
        atomcoord = mol._atom[ia][1]
        # Spin argument doesn't matter but has to give it to use the mol object
        atomicno = element_symbols.index(symb) + 1
        atm1 = gto.M(atom=[[symb, atomcoord]], basis={purssymb:atombasis}, spin=atomicno%2)
        atm2 = gto.M(atom=[[symb, atomcoord]], basis=REF_BASIS, spin=atomicno%2)

        modelfockao = get_model_fock_atom(atm1, atm2, symb)
        atoms_fock.append(modelfockao)

    fock_ao = linalg.block_diag(*atoms_fock)

    del atoms_fock

    return fock_ao

def get_modified_virtuals(mol, mo_energy, mo_coeff):
    """
    Virtual orbitals are not well defined for
    model Fock matrix. This function will modify those virtual orbitals.
    args:
        mol:
            mol object
        mo_energy: list of nao elements
            mo_energy of model fock matrix.
        mo_coeff: nao * nao
            mo_coeff matrix obtained from the model fock matrix
    returns:
        mo_energy: list of nao elements
            mo_energy of the vir orbitals has been modified.
        mo_coeff: nao * nao
            virtual orbitals has been updated for these set of mo_coeff.
    """

    if np.any(mo_energy > VIR_E_CUTOFF):
        # Define the virtual orbitals
        cvir = mo_coeff[:, mo_energy > VIR_E_CUTOFF]

        # Get the kinetic energy intergals and diagonalize them.
        t = mol.intor_symmetric('int1e_kin')
        t_mo = reduce(np.dot, (cvir.T, t, cvir))
        e, u = linalg.eigh(t_mo)
        
        # Shift the energy of virtual orbitals
        e += VIR_E_SHIFT

        # Transform the orbitals basis
        cvir_modified = np.dot(cvir, u)

        # Update the mo_energies and mo_coeffs
        mo_coeff[:, mo_energy > VIR_E_CUTOFF] = cvir_modified
        mo_energy[mo_energy > VIR_E_CUTOFF] = e

    return mo_energy, mo_coeff

def _orthonormalization(mol, fock):
    """
    Diagonalize the fock to obtain the mo_energy and mo_coeff
    args:
        mol:

        fock:
            fock matrix in ao basis
    returns:
        mo_energy:
            list of nao elements
        mo_coeff: nao * nao
            molecular orbital coeffs
    """

    s = mol.intor_symmetric('int1e_ovlp')

    sval, svec = linalg.eigh(s)
    
    lmat = np.dot(svec, np.diag(1/np.sqrt(sval)))
    fockm = reduce(np.dot, (s, fock, s))
    fock = reduce(np.dot, (lmat.T, fockm, lmat))
    
    e, c = linalg.eigh(fock)

    e = e[np.argsort(e)]
    c = c[: , np.argsort(e)]
    c  = lmat @ c

    del fock, s, sval, svec, lmat

    return e, c

def sanity_check_for_orthonormality(mol, mo_coeff):
    """
    This function checks the orthonoramlity for the mo_coeff.
    args:
        mol:
        mo_coeff:
    """
    s = mol.intor_symmetric('int1e_ovlp')
    vtv = reduce(np.dot, (mo_coeff.T, s, mo_coeff))
    assert np.max(abs(vtv-np.eye(s.shape[0]))) < 1e-8, \
    "Orbitals are not orthonormal"

def _finalize(mol):
    '''
    Guessorb module.
    arg:
        mol:
    returns:
        mo_energy: list of nao elements
        mo_coeff:
            mo_coeff matrix
    '''
    modelfock = get_model_fock(mol)
    e, c = _orthonormalization(mol, modelfock)
    mo_energy, mo_coeff = get_modified_virtuals(mol, e, c)
    
    sanity_check_for_orthonormality(mol, mo_coeff)

    return mo_energy, mo_coeff 

# Try to put these functions in a class. and make it pyscf standard.
get_guessorb = _finalize

if __name__ == "__main__":
    mol = gto.M()
    mol.atom='''
    N 0.000000 0.000000 0.000000
    '''
    mol.basis='sto-3g'
    mol.spin = 3
    mol.build()

    mo_energy, mo_coeff = get_guessorb(mol)
    
    from pyscf.tools import molden
    molden.from_mo(mol, 'guessorb.molden', mo_coeff)

