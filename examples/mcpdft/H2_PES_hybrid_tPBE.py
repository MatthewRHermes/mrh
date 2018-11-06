import sys
sys.path.append ('../../..')
import numpy as np
import itertools
from pyscf import gto, scf, mcscf, dft, fci
from mrh.my_pyscf.mcpdft import mcpdft, otfnal

# Examine the effect of "hybridizing" the translated PBE functional using some fraction of CAS "exchange energy"

def cas_22_pt (HHdist, xfnal, cfnal, hybs, basis='sto3g', output=None):
    ''' Don't include 0 in hybs; will do this automatically
    '''
    if output is None:
        output = 'H2_{:.1f}.log'.format (HHdist)
    mol = gto.M (atom = 'H 0 0 0; H 0 0 {:.1f}'.format (HHdist), basis=basis, symmetry=True, output=output, verbose=4)
    hf = scf.RHF (mol)
    hf.kernel ()
    ks = dft.RKS (mol)
    #ks.grids.level = 9
    ks.xc = xfnal + ', ' + cfnal
    e_rks = ks.kernel ()
    ot = otfnal.transfnal (ks)
    mc = mcscf.CASSCF (hf, 2, 2)
    mc.fcisolver = fci.solver (mol, singlet=True, symm=True)
    #mc.fix_spin_(ss=0)
    e_cas = mc.kernel ()[0]
    assert (mc.converged)
    e_pdft = mcpdft.kernel (mc, ot)
    e_hyb = []
    for hyb in hybs:
        ks.xc = '{0:.2f}*HF + {1:.2f}*{2:s}, {3:s}'.format (hyb, 1.0-hyb, xfnal, cfnal)
        e_hyb.append (ks.kernel ())
        ot.otxc = 't{0:.2f}*HF + {1:.2f}*{2:s}, {3:s}'.format (hyb, 1.0-hyb, xfnal, cfnal)
        e_hyb.append (mcpdft.kernel (mc, ot))
    return [e_cas, e_pdft, e_rks] + [e for e in e_hyb]

xfnal = 'XC_GGA_X_PBE'
cfnal = 'XC_GGA_C_PBE'
gga_fnal = 'PBE'
hyb_gga_fnal = 'PBE0'
hybs = [0.2, 0.25, 0.5]
fmt_str = ' '.join (['{:s}' for i in range (4 + 2*len (hybs))])
hybs_str = ['{0:.2f}*HFexcRKS {0:.2f}*CASexcPDFT'.format (hyb).split () for hyb in hybs]
hybs_str = list (itertools.chain (*hybs_str))
line = fmt_str.format ('HHdist', 'CASSCF(2,2)', 'MC-PDFT', 'RKS', *hybs_str)
print (line)
with open ('H2_PES_scan.log', 'w') as f:
    f.write (line + '\n')
    fmt_str = '{:.1f} ' + ' '.join (['{:.8f}' for i in range (3 + 2*len (hybs))])
    for HHdist in np.arange (0.5, 4.01, 0.1):
        line = fmt_str.format (HHdist, *cas_22_pt (HHdist, xfnal, cfnal, hybs))
        print (line)
        f.write (line + '\n')
    f.write ('\n')

def ref_pt (HHdist, gga_fnal, hyb_gga_fnal, basis='sto3g', output=None):
    if output is None:
        output = 'H2_{:.1f}_ref.log'.format (HHdist)
    mol = gto.M (atom = 'H 0 0 0; H 0 0 {:.1f}'.format (HHdist), basis=basis, symmetry=True, output=output, verbose=4)
    hf = scf.RHF (mol)
    hf.kernel ()
    ks = dft.RKS (mol)
    #ks.grids.level = 9
    ks.xc = gga_fnal
    e_gga = ks.kernel ()
    ks.xc = hyb_gga_fnal
    e_hyb_gga = ks.kernel ()
    ks.xc = gga_fnal
    ot = otfnal.transfnal (ks)
    mc = mcscf.CASSCF (hf, 2, 2)
    mc.fcisolver = fci.solver (mol, singlet=True, symm=True)
    #mc.fix_spin_(ss=0)
    e_cas = mc.kernel ()[0]
    assert (mc.converged)
    e_pdft = mcpdft.kernel (mc, ot)
    return e_cas, e_gga, e_hyb_gga, e_pdft
    
line = 'HHdist CASSCF(2,2) RKS(GGA) RKS(HGGA) MC-PDFT(GGA)'
print (line)
with open ('H2_PES_scan.log', 'a') as f:
    f.write (line + '\n')
    fmt_str = '{:.1f} ' + ' '.join (['{:.8f}' for i in range (4)])
    for HHdist in np.arange (0.5, 4.01, 0.1):
        line = fmt_str.format (HHdist, *ref_pt (HHdist, gga_fnal, hyb_gga_fnal))
        print (line)
        f.write (line + '\n')

