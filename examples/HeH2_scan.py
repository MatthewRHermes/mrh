# usage: 
# > python HeH2_scan.py [options]
# where options can include
#    -> dist=x1,x2,x3,...: do calculations at H-H distance x1, x2, x3 in angstroms instead of the default
#    -> casci: for a single-embedding CAS-DMET calculation with only the N2 fragment
#    -> doDET: do a DET calculation rather than a DMET calculation 
#    -> mp2: to use MP2, rather than RHF, to solve the impurity problem for the two methyl fragments
#    -> ofc_emb: to use MRH's "other-fragment core embedding" rather than standard DMET

import sys,re
from mrh.my_dmet import localintegrals, dmet
from mrh.my_dmet.fragments import make_fragment_atom_list, make_fragment_orb_list
import numpy as np
import HeH2_struct
from pyscf import scf
localization = 'meta_lowdin'        # 'iao' or 'meta_lowdin' or 'boys'
casci_energy_formula = False     # CASCI or DMET energy formula
CC_E_TYPE = 'LAMBDA'
doDET = False
basis = '6-31g'
helium_method = "RHF"
dist_re = re.compile ('[\d\.]+')

distlist = np.concatenate ((np.arange (0.5, 1.05, 0.1), np.arange (1.6, 5.7, 1.0)))
ofc_embedding = False
noselfconsistent = False

for iargv in sys.argv[1:]:
    if iargv == "casci":
        casci_energy_formula = True
        CC_E_TYPE = 'CASCI'
    elif iargv == "doDET":
        doDET = True
    elif iargv == "ofc_emb":
        ofc_embedding = True
    elif iargv == "mp2":
        helium_method="MP2"
#        noselfconsistent=True
    elif iargv[:5] == 'dist=':
        distlist = [float (x) for x in dist_re.findall (iargv)]

for distance in distlist: # H-H distance
    print ('--------------------------------------------')
    print ('---- HeH-H at',distance,'angstroms  --------')
    print ('--------------------------------------------')    
    # Set up the scf
    mol = HeH2_struct.structure (distance, basis)
    mf = scf.RHF (mol)
    mf.verbose = 3
    mf.scf ()
    norbs_tot = mol.nao_nr ()
    
    # Set up the integrals
    myInts = localintegrals.localintegrals(mf, range(norbs_tot), localization)
    myInts.molden( 'HeH2.molden' )
    
    # Build fragments
    H2_from_atomlist = make_fragment_atom_list (myInts, list (range(2)), 'CASSCF(2,2)', name="H2")
    He_from_atomlist = make_fragment_atom_list (myInts, [2], helium_method, name="He")
    fraglist = [H2_from_atomlist] if casci_energy_formula else [H2_from_atomlist, He_from_atomlist]
    
    # Build class
    HeH2dmet = dmet (myInts, fraglist, doDET=doDET, CC_E_TYPE=CC_E_TYPE, ofc_embedding=ofc_embedding, debug_energy=True, noselfconsistent=noselfconsistent)

    # Do calculation
    energy_result = HeH2dmet.doselfconsistent ()
    print ("----Energy at {0:.1f} angstrom: {1}".format (distance, energy_result))

