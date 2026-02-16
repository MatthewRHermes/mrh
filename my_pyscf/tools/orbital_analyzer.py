#!/usr/bin/env python
#
# Author: Valay Agarawal <valayagarawal@gmail.com>
#

'''
Recommended to use avas selected orbitals
Basic recommender for selecting active spaces based on avas active space.
Also gives a description of orbitals shapes
'''
import numpy as np

def describe_orbitals(mol, mo, with_recommendation = False, avas_strings = None,num_contributors = 4):
  '''
    Input: 
      Args:
        mol: molecule object
        mo: mo_coeff 
      Kwargs: 
        with_recommendations: (True/False) Tells whether you want recommended orbitals
        avas_strings: (list of strings) ['O 2px', 'O@2 2pz']
        num_contributors: 3-4 are good enough. 
          More number of contributors: Noisy
          Less number of contributors: Fewer orbitals
    Output: 
        No output if no recommendations required
        If recommendations are needed, returns a list of orbitals
  ''' 
  
  #TODO: 1. Turn printing to logging
  print("All possible mo coefficients")
  all_aos = mol.ao_labels()
  #print(all_aos)
  assert mo.shape[0] == mol.nao, "size incorrect"
  if mo.shape[1] == mol.nao: ncore = 0
  else: ncore = 'ncore'

  print(f"In:{mo.shape[1]} orbitals")
  print(f"This starts indexing from {ncore}")
  if with_recommendation: 
    if avas_strings is None: 
      print("if you want recommendations, you need to give me an avas string that denotes orbitals you want")
      exit()
    total_recommended_orbitals = 0
    for string in avas_strings:
      total_recommended_orbitals += len([orb_label for orb_label in all_aos if string in orb_label])
    print("Total recommended orbitals", total_recommended_orbitals)
    recommendations = []
  else: 
    print("Describing orbitals")
  list_of_orbs = []
  for idx in range(mo.shape[1]):
    orb = abs(mo[:,idx])
    ind = np.argpartition(orb,-num_contributors)[-num_contributors:]
    coeffs=orb[ind]
    tmp_ind_sorted = np.argsort(coeffs)[::-1] #largest to smallest
    ind_sorted = ind[tmp_ind_sorted]
    major_orbs = np.asarray(mol.ao_labels())[ind_sorted]
    major_coeffs = (mo[:,idx])[ind_sorted]
    line=f"Orbital {idx}: "
     
    #TODO: 2. Make it a while loop with "popping" logic that removes orbitals that have the strings, but need a lot more work on that
    for m_orb, m_coeff in zip(major_orbs, major_coeffs):
      line+=f'{m_orb} {round(m_coeff,3)} '
    if with_recommendation:
      orb_to_be_added = 0
      for target_orbital in avas_strings:
        orb_to_be_added += len([m_orb for m_orb in major_orbs if target_orbital in m_orb])
      if orb_to_be_added: recommendations.append(line)
      list_of_orbs.append(idx)
      
    #if with_recommendation is None: print(line)
    #print(line)
  if with_recommendation:
    print(f"Total recommendations needed: {total_recommended_orbitals}, actual: {len(recommendations)}")
    print("This is my recommendation")
    
    for line in recommendations:
      print(line)
    return list_of_orbs
  else:
    return None
  


if __name__ == "__main__":
  import pyscf
  from pyscf import gto, scf,mcscf
  from pyscf.mcscf import avas
  mol = gto.M(atom = "H 0 0 0; H 0 0 1;",basis="sto3g")

  mol = gto.M(
    atom = '''O 0 0 0; O@2 0 1 0''',
    basis = {'default': '631g','O@2':'ccpvdz'})
  mf = scf.RHF(mol).density_fit().run()
  ncas, nelecas, guess_mo_coeff = avas.kernel(mf,['O 2p', 'O@2 2p'],minao = mol.basis)
  mc_test = mcscf.CASCI (mf, ncas, nelecas)
  ncas_orbs = guess_mo_coeff[:, mc_test.ncore:mc_test.ncore + ncas]
  describe_orbitals(mol, guess_mo_coeff, with_recommendation=True,avas_strings =['O 2py', 'O@2 2py'])
  describe_orbitals(mol, ncas_orbs, with_recommendation=True,avas_strings =['O 2py', 'O@2 2py'])
