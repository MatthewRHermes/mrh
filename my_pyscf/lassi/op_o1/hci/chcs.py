import numpy as np
from pyscf.lib import param
from mrh.my_pyscf.lassi.op_o1.hci.schcs import ContractHamCI_SHS

# This is a quick-and-dirty kludge that is only meant to work for 2-fragment systems

class ContractHamCI_CHS (ContractHamCI_SHS):
    def __init__(self, las, ints, nlas, lroots, h0, h1, h2, si_ket,
                 mask_bra_space=None, mask_ket_space=None, pt_order=None, do_pt_order=None,
                 add_transpose=False, accum=None, log=None, max_memory=param.MAX_MEMORY,
                 dtype=np.float64):
        si_bra = fake_si_bra (lroots, si_ket, mask_bra_space)
        ContractHamCI_SHS.__init__(self, las, ints, nlas, lroots, h0, h1, h2, si_bra, si_ket,
                                   mask_bra_space=mask_bra_space,
                                   mask_ket_space=mask_ket_space,
                                   pt_order=pt_order,
                                   do_pt_order=do_pt_order,
                                   add_transpose=add_transpose,
                                   accum=accum, log=log, max_memory=max_memory, dtype=dtype)

def fake_si_bra (lroots, si_ket, mask_bra_space):
    nfrags, nroots = lroots.shape
    assert (nfrags==2)
    lroots_bra = lroots[:,mask_bra_space]
    assert (np.all (lroots_bra[0]==lroots_bra[1]))
    nprods = lroots.prod (0)
    offs = np.cumsum (nprods) - nprods
    assert (si_ket.shape[0]==nprods.sum())
    lroots_bra = lroots[:,mask_bra_space]
    si_bra = np.zeros_like (si_ket)
    for iroot in mask_bra_space:
        i = offs[iroot]
        lr = lroots[0,iroot]
        j = i + nprods[iroot]
        assert (nprods[iroot]==lr*lr)
        si_bra[i:j,:] = np.eye (lr)[:,:,None].reshape (lr*lr,-1)
    return si_bra


