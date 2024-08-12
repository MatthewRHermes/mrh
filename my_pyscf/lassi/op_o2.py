import time
import numpy as np
from scipy import linalg
from pyscf.lib import logger
from mrh.my_pyscf.lassi import op_o1

make_stdm12s = op_o1.make_stdm12s
ham = op_o1.ham
contract_ham_ci = op_o1.contract_ham_ci
fermion_frag_shuffle = op_o1.fermion_frag_shuffle

class LRRDMint (op_o1.LRRDMint):
    __doc__ = op_o1.LRRDMint.__doc__ + '''

    op_o2 reimplementation: get rid of outer products! This will take a while...
    '''

    def get_single_rootspace_sivec (self, iroot):
        '''A single-rootspace slice of the SI vectors, reshaped to expose the lroots.

        Args:
            iroot: integer
                Rootspace index

        Returns:
            sivec: col-major ndarray of shape (lroots[0,iroot], lroots[1,iroot], ...,
                                               nroots_si)
                SI vectors
        '''
        i, j = self.offs_lroots[iroot]
        vecshape = list (self.lroots[:,iroot]) + [self.nroots_si,]
        return self.si[i:j,:].reshape (vecshape, order='F')

    def get_frag_transposed_sivec (self, iroot, *inv):
        '''A single-rootspace slice of the SI vectors, transposed so that involved fragments
        are slower-moving

        Args:
            iroot: integer
                Rootspace index
            *inv: integers 
                Indices of nonspectator fragments

        Returns:
            sivec: col-major ndarray of shape (ncols, nrows, nroots_si)
                SI vectors with the first dimension iterating over states of fragments not in
                inv and the second dimension iterating over states of fragments in inv 
        '''
        axesorder = [i for i in range (self.nfrags) if not (i in inv)] + list(inv) + [self.nfrags,]
        sivec = self.get_single_rootspace_sivec (iroot).transpose (*axesorder)
        nprods = np.prod (self.lroots[:,iroot])
        nrows = np.prod (self.lroots[inv,iroot])
        ncols = nprods // nrows
        return np.asfortranarray (sivec).reshape ((ncols, nrows, self.nroots_si), order='F')

    def get_fdms (self, rbra, rket, *inv, _lowertri=True):
        '''Get the n-fragment density matrices for the fragments identified by inv in the bra and
        spaces given by rbra and rket, summing over nonunique excitations

        Args: 
            rbra: integer
                Index of bra rootspace for which to prepare the current cache.
            rket: integer
                Index of ket rootspace for which to prepare the current cache.
            *inv: integers 
                Indices of nonspectator fragments

        Returns:
            fdm : col-major ndarray of shape (lroots[inv[0],rbra], lroots[inv[0],rket],
                                              lroots[inv[1],rbra], lroots[inv[1],rket],
                                               ..., nroots_si)
                len(inv)-fragment reduced density matrix
        '''
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        key = tuple ((rbra,rket)) + inv
        braket_table = self.nonuniq_exc[key]
        fdm = 0
        for rbra1, rket1 in braket_table:
            b, k, o = self._get_spec_addr_ovlp_1space (rbra1, rket1, *inv, _lowertri=_lowertri)
            # Numpy pads array dimension to the left, so transpose
            sibra = self.get_frag_transposed_sivec (rbra, *inv)[b,:,:].T * o
            siket = self.get_frag_transposed_sivec (rket, *inv)[k,:,:].T
            fdm += np.stack ([np.dot (b, k.T) for b, k in zip (sibra, siket)], axis=-1)
        fdm = np.asfortranarray (fdm)
        newshape = list (self.lroots[inv,rbra]) + list (self.lroots[inv,rket]) + [self.nroots_si,]
        fdm = fdm.reshape (newshape, order='F')
        axesorder = sum ([[i, i+len(inv)] for i in range (len(inv))], []) + [2*len(inv),]
        fdm = fdm.transpose (*axesorder)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_o, self.dw_o = self.dt_o + dt, self.dw_o + dw
        return np.asfortranarray (fdm)

    def _get_spec_addr_ovlp_1space (self, rbra, rket, *inv, _lowertri=True):
        '''Obtain the integer indices and overlap*permutation factors for all pairs of model states
        in the same rootspaces as bra, ket for which a specified list of nonspectator fragments are
        also in same state that they are in a provided input pair bra, ket.

        Args:
            rbra: integer
                Index of a rootspace
            rket: integer
                Index of a rootspace
            *inv: integers
                Indices of nonspectator fragments.

        Returns:
            bra_rng: ndarray of integers
                Indices corresponding to nonzero overlap factors for the ENVs of inv only
            ket_rng: ndarray of integers
                Indices corresponding to nonzero overlap factors for the ENVs of inv only
            o: ndarray of floats
                Overlap * permutation factors (cf. get_ovlp_fac) corresponding to the interactions
                bra_rng, ket_rng.
        '''
        inv = list (set (inv))
        fac = self.spin_shuffle[rbra] * self.spin_shuffle[rket]
        fac *= fermion_frag_shuffle (self.nelec_rf[rbra], inv)
        fac *= fermion_frag_shuffle (self.nelec_rf[rket], inv)
        spec = np.ones (self.nfrags, dtype=bool)
        for i in inv: spec[i] = False
        spec = np.where (spec)[0]
        specints = [self.ints[i] for i in spec]
        o = fac * np.ones ((1,1), dtype=self.dtype)
        for i in specints:
            b, k = i.unique_root[rbra], i.unique_root[rket]
            o = np.multiply.outer (i.ovlp[b][k], o).transpose (0,2,1,3)
            o = o.reshape (o.shape[0]*o.shape[1], o.shape[2]*o.shape[3])
        idx = np.abs(o) > 1e-8
        if _lowertri and (rbra==rket): # not bra==ket because _loop_lroots_ doesn't restrict to tril
            o[np.diag_indices_from (o)] *= 0.5
            idx[np.triu_indices_from (idx, k=1)] = False
        o = o[idx]
        bra_rng, ket_rng = np.where (idx)
        return bra_rng, ket_rng, o
                

def get_sdm1_maker (las, ci, nelec_frs, si, **kwargs):
    log = logger.new_logger (las, las.verbose)
    nlas = las.ncas_sub
    ncas = las.ncas 
    nroots_si = si.shape[-1]
    max_memory = getattr (las, 'max_memory', las.mol.max_memory)
    dtype = ci[0][0].dtype 
        
    # First pass: single-fragment intermediates
    hopping_index, ints, lroots = op_o1.make_ints (las, ci, nelec_frs)
    nstates = np.sum (np.prod (lroots, axis=0))
        
    # Second pass: upper-triangle
    outerprod = LRRDMint (ints, nlas, hopping_index, lroots, si, dtype=dtype,
                          max_memory=max_memory, log=log)
    def make_sdm1 (iroot, ifrag):
        return outerprod.get_fdms (iroot, iroot, ifrag, _lowertri=False)
    return make_sdm1



        

