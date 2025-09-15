import numpy as np
from mrh.my_pyscf.fci import spin_op

def _get_highm_civecs (cibra, ciket, norb, nelec, dnelec, smult_bra, smult_ket):
    if smult_bra is None or smult_ket is None:
        return cibra, ciket, nelec
    nelec_ket = nelec
    nelec_bra = (nelec[0]+dnelec[0], nelec[1]+dnelec[1])
    cibra = spin_op.mup (cibra, norb, nelec_bra, smult_bra)
    ciket = spin_op.mup (ciket, norb, nelec_ket, smult_ket)
    nelec_bra = sum (nelec_bra)
    nelec_ket = sum (nelec_ket)
    dspin_op = dnelec[0]-dnelec[1]
    spin_ket = min (smult_ket-1, smult_bra-1-dspin_op)
    spin_bra = spin_ket + dspin_op
    nelec_bra = ((nelec_bra + spin_bra)//2, (nelec_bra-spin_bra)//2)
    nelec_ket = ((nelec_ket + spin_ket)//2, (nelec_ket-spin_ket)//2)
    cibra = spin_op.mdown (cibra, norb, nelec_bra, smult_bra)
    ciket = spin_op.mdown (ciket, norb, nelec_ket, smult_ket)
    return cibra, ciket, nelec_ket

