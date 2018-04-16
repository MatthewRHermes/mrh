'''
    QC-DMET: a python implementation of density matrix embedding theory for ab initio quantum chemistry
    Copyright (C) 2015 Sebastian Wouters
    
    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.
    
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
'''

from . import localintegrals
import mrh.my_dmet.rhf
import numpy as np
import ctypes
from ctypes.util import find_library
from mrh.util.basis import represent_operator_in_basis
lib_qcdmet = ctypes.CDLL('/home/gagliard/herme068/Apps/QC-DMET/lib/libqcdmet.so')
#lib_qcdmet = ctypes.CDLL(find_library('qcdmet'))

class qcdmethelper:

    def __init__( self, theLocalIntegrals, list_H1, altcf, minFunc ):
    

        self.locints = theLocalIntegrals
        assert( self.locints.Nelec % 2 == 0 )
        self.numPairs = int (self.locints.Nelec / 2)
        self.altcf = altcf
        self.minFunc = None

        if self.altcf:
            assert (minFunc == 'OEI' or minFunc == 'FOCK_INIT')
            self.minFunc = minFunc
        
        # Variables for c gradient calculation
        self.list_H1 = list_H1
        H1start, H1row, H1col = self.convertH1sparse()
        self.H1start = H1start
        self.H1row = H1row
        self.H1col = H1col
        self.Nterms = len( self.H1start ) - 1

        # Variables related to the 1RDM calculations, possibly dependent on some frozen degrees of freedom
        self.OEI_wrk = None
        self.OEI_loc = None
        self.oneRDMfrz_loc = None
        self.TEI_wrk = None
        self.ao2wrk = None
        self.numPairs_wrk = self.numPairs
        self.loc2wrk = None
        
    def convertH1sparse( self ):
    
        H1start = []
        H1row   = []
        H1col   = []
        H1start.append( 0 )
        totalsize = 0
        for count in range( len( self.list_H1 ) ):
            rowco, colco = np.where( self.list_H1[count] == 1 )
            totalsize += len( rowco )
            H1start.append( totalsize )
            for count2 in range( len( rowco ) ):
                H1row.append( rowco[ count2 ] )
                H1col.append( colco[ count2 ] )
        H1start = np.array( H1start, dtype=ctypes.c_int )
        H1row   = np.array( H1row,   dtype=ctypes.c_int )
        H1col   = np.array( H1col,   dtype=ctypes.c_int )
        return ( H1start, H1row, H1col )

    def init_oei_for_umat_cycle (self):
        if ((self.loc2wrk==None) != (self.oneRDMfrz_loc==None)) or ((self.loc2wrk==None) != (self.nelec_frz==None)):
            raise RuntimeError("Set all of helper.loc2wrk, helper.oneRDMfrz_loc, and helper.nelec_frz or none of them!")
        self.numPairs_wrk = self.numPairs - (int (round (self.nelec_frz)) // 2) if self.nelec_frz else self.numPairs
        self.OEI_wrk, self.TEI_wrk, self.ao2wrk = self.OEI_for_construct1RDM_loc ()
        self.OEI_loc = represent_operator_in_basis (self.OEI_wrk, loc2wrk.T) if self.loc2wrk else self.OEI_wrk

    def OEI_for_construct1RDM_loc ( self ):
        use_OEI = (self.altcf and self.minFunc == 'OEI')

        TEI_wrk = None
        ao2wrk = None
        if self.locints.ERIinMem: 
            TEI_wrk = self.locints.loc_tei ()
        else:
            ao2wrk = self.locints.ao2loc

        if self.loc2wrk == None:
            GOCK_wrk = self.locints.loc_oei () if use_OEI else self.locints.loc_rhf_fock ()
            return GOCK_wrk, TEI_wrk, ao2wrk

        GOCK_wrk = represent_operator_in_basis (self.locints.loc_rhf_fock_bis (self.oneRDMfrz_loc), self.loc2wrk)
        if use_OEI:
            return GOCK_wrk, TEI_wrk, ao2wrk

        oneRDM_wrk = self.construct1RDM_base (GOCK_wrk, numPairs_wrk)
        if ( self.locints.ERIinMEM == True ):
            TEI_wrk = self.locints.dmet_tei (self.loc2wrk, self.loc2wrk.shape[1]) 
            oneRDM_wrk = rhf.solve_ERI( GOCK_wrk, TEI_wrk, oneRDM_wrk, numPairs_wrk )
        else:
            ao2wrk = np.dot (self.locints.ao2loc, self.loc2wrk) 
            oneRDM_wrk = rhf.solve_JK( GOCK_wrk, self.locints.mol, ao2wrk, oneRDM_wrk, numPairs_wrk )
        oneRDM_loc = represent_operator_in_basis (oneRDM_wrk, self.loc2wrk.T) + self.oneRDMfrz_loc
        FOCKINIT_loc = self.locints.loc_rhf_fock_bis (oneRDM_loc)
        FOCKINIT_wrk = represent_operator_in_basis (FOCKINIT_loc, self.loc2wrk)
        return FOCKINIT_wrk, TEI_wrk, ao2wrk
        
    def construct1RDM_loc( self, doSCF, umat_loc):
        umat_wrk = represent_operator_in_basis (umat_loc, self.loc2wrk) if self.loc2wrk else umat_loc
        OEI_plus_umat = self.OEI_wrk + umat_wrk
        oneRDM_wrk = self.construct1RDM_base( OEI_plus_umat, self.numPairs_wrk )
        if ( doSCF == True ):
            if ( self.locints.ERIinMEM == True ):
                oneRDM_wrk = rhf.solve_ERI( OEI_plus_umat, self.TEI_wrk, oneRDM_wrk, self.numPairs_wrk )
            else:
                oneRDM_wrk = rhf.solve_JK( OEI_plus_umat, self.locints.mol, self.ao2wrk, oneRDM_wrk, self.numPairs_wrk )
        oneRDM_loc = represent_operator_in_basis (oneRDM_wrk, self.loc2wrk.T) + self.oneRDMfrz_loc if self.loc2wrk else oneRDM_wrk
        return oneRDM_loc

    def construct1RDM_response( self, doSCF, umat_loc, loc2rspwork ):
        if doSCF:
            oneRDM_loc = self.construct1RDM_loc (doSCF, umat_loc)
            OEI_plus_umat = self.locints.loc_rhf_fock_bis( oneRDM_loc ) + umat_loc
        else:
            OEI_plus_umat = self.OEI_loc + umat_loc
        
        # MRH: only the derivative of the constructed 1RDM ("gamma_D" in my notes) is necessary here, not the wma part of the 1RDM
        # MRH: However I'm not 100% sure that this can be done in a complete basis because what if the derivatives tell that part of the 1RDM
        #      to go into the wma part???  How does rhf_response work????
        # MRH: I ~think~ I don't have to do anything to the rhf_response part; I just have to make sure that I give it the right OEI ("GOCK")
        # MRH: since the complete derivative gets multiplied by a component which is zero in the wma by construction, there shouldn't be a problem
        #      as long as I set up the error vector correctly 
        rdm_deriv_rot = np.ones( [ self.locints.Norbs * self.locints.Norbs * self.Nterms ], dtype=ctypes.c_double )
        if ( loc2rspwork != None ):
            OEI_plus_umat = np.dot( np.dot( loc2rspwork.T, OEI_plus_umat ), loc2rspwork )
        OEI = np.array( OEI_plus_umat.reshape( (self.locints.Norbs * self.locints.Norbs) ), dtype=ctypes.c_double )
        
        lib_qcdmet.rhf_response( ctypes.c_int( self.locints.Norbs ),
                                 ctypes.c_int( self.Nterms ),
                                 ctypes.c_int( self.numPairs ),
                                 self.H1start.ctypes.data_as( ctypes.c_void_p ),
                                 self.H1row.ctypes.data_as( ctypes.c_void_p ),
                                 self.H1col.ctypes.data_as( ctypes.c_void_p ),
                                 OEI.ctypes.data_as( ctypes.c_void_p ),
                                 rdm_deriv_rot.ctypes.data_as( ctypes.c_void_p ) )
        
        rdm_deriv_rot = rdm_deriv_rot.reshape( (self.Nterms, self.locints.Norbs, self.locints.Norbs), order='C' )
        return rdm_deriv_rot
        
    def construct1RDM_base( self, OEI, myNumPairs ):
    
        eigenvals, eigenvecs = np.linalg.eigh( OEI ) # Does not guarantee sorted eigenvectors!
        idx = eigenvals.argsort()
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:,idx]
        OneDM = 2 * np.dot( eigenvecs[:,:myNumPairs] , eigenvecs[:,:myNumPairs].T )
        #print "SP gap =", eigenvals[myNumPairs] - eigenvals[myNumPairs-1]
        return OneDM
        
    def constructbath( self, OneDM, impurityOrbs, numBathOrbs, threshold=1e-13 ):
    
        embeddingOrbs = 1 - impurityOrbs
        embeddingOrbs = np.matrix( embeddingOrbs )
        if (embeddingOrbs.shape[0] > 1):
            embeddingOrbs = embeddingOrbs.T # Now certainly row-like matrix (shape = 1 x len(vector))
        isEmbedding = np.dot( embeddingOrbs.T , embeddingOrbs ) == 1
        numEmbedOrbs = np.sum( embeddingOrbs )
        embedding1RDM = np.reshape( OneDM[ isEmbedding ], ( numEmbedOrbs , numEmbedOrbs ) )

        numImpOrbs   = np.sum( impurityOrbs )
        numTotalOrbs = len( impurityOrbs )
        
        eigenvals, eigenvecs = np.linalg.eigh( embedding1RDM )
        idx = np.maximum( -eigenvals, eigenvals - 2.0 ).argsort() # Occupation numbers closest to 1 come first
        tokeep = np.sum( -np.maximum( -eigenvals, eigenvals - 2.0 )[idx] > threshold )
        if ( tokeep < numBathOrbs ):
            print("DMET::constructbath : Throwing out", numBathOrbs - tokeep, "orbitals which are within", threshold, "of 0 or 2.")
        numBathOrbs = min(np.sum( tokeep ), numBathOrbs)
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:,idx]
        pureEnvironEigVals = -eigenvals[numBathOrbs:]
        pureEnvironEigVecs = eigenvecs[:,numBathOrbs:]
        idx = pureEnvironEigVals.argsort()
        eigenvecs[:,numBathOrbs:] = pureEnvironEigVecs[:,idx]
        pureEnvironEigVals = -pureEnvironEigVals[idx]
        coreOccupations = np.hstack(( np.zeros([ numImpOrbs + numBathOrbs ]), pureEnvironEigVals ))
    
        for counter in range(0, numImpOrbs):
            eigenvecs = np.insert(eigenvecs, counter, 0.0, axis=1) #Stack columns with zeros in the beginning
        counter = 0
        for counter2 in range(0, numTotalOrbs):
            if ( impurityOrbs[counter2] ):
                eigenvecs = np.insert(eigenvecs, counter2, 0.0, axis=0) #Stack rows with zeros on locations of the impurity orbitals
                eigenvecs[counter2, counter] = 1.0
                counter += 1
        assert( counter == numImpOrbs )
    
        # Orthonormality is guaranteed due to (1) stacking with zeros and (2) orthonormality eigenvecs for symmetric matrix
        assert( np.linalg.norm( np.dot(eigenvecs.T, eigenvecs) - np.identity(numTotalOrbs) ) < 1e-12 )

        # eigenvecs[ : , 0:numImpOrbs ]                      = impurity orbitals
        # eigenvecs[ : , numImpOrbs:numImpOrbs+numBathOrbs ] = bath orbitals
        # eigenvecs[ : , numImpOrbs+numBathOrbs: ]           = pure environment orbitals in decreasing order of occupation number
        return ( numBathOrbs, eigenvecs, coreOccupations )
        
