# usage: 
# > python Me2N2-DMET-scan.py dist [options]
# where dist is the N-N distance in angstrom and options can include
#    -> casci: for a single-embedding CAS-DMET calculation with only the N2 fragment
#    -> trunc: to force the N2 fragment to have only two bath orbitals
#    -> mp2: to use MP2, rather than RHF, to solve the impurity problem for the two methyl fragments
#    -> ofc_emb: to use MRH's "other-fragment core embedding" rather than standard DMET

import sys,re
sys.path.append ('../my_dmet')
import localintegrals, dmet, fragments
from fragments import make_fragment_atom_list, make_fragment_orb_list
from pyscf import scf
import numpy as np
import Me2N2_struct

#############
#   Input   #
#############              
localization = 'meta_lowdin'                # 'iao' or 'meta_lowdin' or 'boys'
one_bath_orb_per_bond = False        # Sun & Chan, JCTC 10, 3784 (2014) [ http://dx.doi.org/10.1021/ct500512f ]
doDET = True
ofc_embedding = False
CC_E_TYPE = 'LAMBDA'
basis = '6-31g' 
me_method = 'RHF'
active_space_selection = np.loadtxt ("Me2N2_active_space.dat", dtype=int).tolist ()
ints = re.compile ("[0-9]+")
print (sys.argv)
distance = float (sys.argv[1])
CASlist = np.empty (0, dtype=int)
for iargv in sys.argv[2:]:
    if iargv == 'casci':
        CC_E_TYPE = 'CASCI'
        print ("Doing CASCI calculation")
    elif iargv == 'trunc':
        one_bath_orb_per_bond = True
        print ("Note: one bath orbital per bond specified; per Hung's calculations this is only supposed to affect N2")
    elif iargv == 'ofc_emb':
        ofc_embedding = True
    elif iargv == 'mp2':
        me_method = 'MP2'
    else:
        CASlist = [int (i) for i in ints.findall (iargv)] 
        print ("Manual CASlist: {0}".format (CASlist))

if one_bath_orb_per_bond:
    CASlist = np.empty (0, dtype=int)
elif len (CASlist) == 0:
    idx = int (round ((distance-1) * 10))
    CASlist = active_space_selection[idx]


print ('-----------------------------------------------')
print ('----  Me-N=N-Me at',distance,'angstroms  --------')
print ('-----------------------------------------------')    
mol = Me2N2_struct.structure( distance, basis)
mf = scf.RHF( mol )
mf.verbose = 3
mf.scf()
norbs_tot = mol.nao_nr ()

# Set up the integrals
myInts = localintegrals.localintegrals(mf, range(norbs_tot), localization)
myInts.molden( 'Me2N2.molden' )

# Build fragments from atom list
N2 = make_fragment_atom_list (myInts, list (range(2)), 'CASSCF(4,4)', name="N2", active_orb_list=CASlist)
if one_bath_orb_per_bond:
    N2.norbs_bath_max = 2
    N2.idempotize_thresh = 0.25
Me1 = make_fragment_atom_list (myInts, list (range(2,6)), me_method, name='Me1')
Me2 = make_fragment_atom_list (myInts, list (range(6,10)), me_method, name='Me2')
fraglist = [N2] if CC_E_TYPE == 'CASCI' else [N2, Me1, Me2]

# Testing class creator
Me2N2_dmet = dmet (myInts, fraglist, doDET=doDET, CC_E_TYPE=CC_E_TYPE, ofc_embedding=ofc_embedding, debug_energy=False)

# Testing calculation
energy_result = Me2N2_dmet.doselfconsistent ()

print ("----Energy at {0:.1f} angstrom: {1}".format (distance, energy_result))

   
