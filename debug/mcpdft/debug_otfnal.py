from pyscf import gto, scf
from pyscf.dft.libxc import XC_KEYS, is_meta_gga, hybrid_coeff, rsh_coeff
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf.mcpdft import _libxc

mol = gto.M (atom = 'H 0 0 0; H 1 0 0', basis='sto-3g', verbose=0, output='/dev/null')
mf = scf.RHF (mol).run ()
mc = mcpdft.CASSCF (mf, 'tPBE', 2, 2, grids_level=1).run ()

XC_NAMES = _libxc.XC_ALIAS_KEYS.union (XC_KEYS)
print (len (XC_KEYS), len (_libxc.XC_ALIAS_KEYS), len (XC_NAMES))

mc.energy_tot (otxc='tGGA_XC_THFL')

i = 0
for xc_name in XC_NAMES:
    if xc_name in ('GGA_X_LB','GGA_X_LBM','LDA_XC_TIH'):
        # Causes LibXC to send a kill message
        continue
    if is_meta_gga (xc_name): continue
    ot_name = 't'+xc_name
    e_tot = e_nuc = None
    try:
        e_tot, e_ot = mc.energy_tot (otxc=ot_name)
        e_nuc, e_1e, e_coul, e_otx, e_otc, e_mcwfn = mc.get_energy_decomposition (otxc=ot_name)
        assert (abs(e_otx+e_otc-e_ot)<1e-10)
        assert (abs(e_nuc+e_1e+e_coul+e_otx+e_otc-e_tot)<1e-10)
    except Exception as e:
        if xc_name.startswith ('HYB') and str (e).startswith ('Aliased'):
            pass
        elif ('_' in xc_name and xc_name.split ('_')[1]=='XC' and
              str (e).startswith ('LibXC built-in X+C')):
            pass
        elif ('_' in xc_name and xc_name.split ('_')[1]=='K' and
              str (e).startswith ('Kinetic')):
            pass
        else:
            print ("\n{} {}".format (i, ot_name))
            i += 1
            print (e.__class__.__name__)
            print (str (e))
            if e_tot is not None: print (e_tot, e_ot)
            if e_nuc is not None: print (e_nuc, e_1e, e_coul, e_otx, e_otc, e_mcwfn)
            if str (e).startswith ('Aliased'):
                print (hybrid_coeff (xc_name), rsh_coeff (xc_name)[0])


