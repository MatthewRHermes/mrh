from pyscf import gto, scf, mcscf
from mrh.my_pyscf.gto import get_ano_rcc_basis

'''
Generalizing the use the ANO-RCC basis sets in PySCF
1. Different ANO-RCC basis sets for different elements
2. Using ANO-RCC-V*ZP with mix-basis sets. (As in ANO-RCC for one element and cc-pVDZ-DK for another)
3. Also, works with 'default' keyword in basis dictionary.

Available basis sets: 
1. 'ANO-RCC-MB'
2. 'ANO-RCC-VDZP'
3. 'ANO-RCC-VTZP'
4. 'ANO-RCC-VQZP'

Required: mol object with atoms and coordinates defined.
Returns: a dictionary that can be directly used as basis argument in mol object.
'''

# Example-1
mol1 = gto.M(atom='Fe 0 0 0; Xe 0 0 10; Cd 0 0 20', basis='ano', charge=2, verbose=0)
basis = get_ano_rcc_basis (mol1, 'VTZP')
print(basis)
mol1.basis = basis
mol1.build ()

# Example-2: Using mix-basis sets
mol1 = gto.Mole(atom='Fe 0 0 0; Xe 0 0 10; Cd 0 0 20', charge=2, verbose=0)
basis = get_ano_rcc_basis (mol1, {'Fe':'ANO-RCC-VTZP', 'Cd':'ANO-RCC-VDZP', 'Xe':'ANO-RCC-MB'})
print(basis)
mol1.basis = basis
mol1.build ()

# Example-3: Using 'default' keyword in basis dictionary
mol1 = gto.Mole(atom='Fe 0 0 0; He 0 0 10; Mg 0 0 20', charge=2, verbose=0)
basis = get_ano_rcc_basis (mol1, {'Fe':'ANO-RCC-VQZP', 'default':'cc-pvdz-dk'})
print(basis)
mol1.basis = basis
mol1.build ()
