from pyscf import gto, scf, mcscf, lib

mol = gto.M (atom='Li 0 0 0\nH 1.5 0 0', basis='sto-3g', symmetry=True,
             output=__file__+'.log', verbose=lib.logger.INFO)
mf = scf.RHF (mol).run ()
print (getattr (mf.mo_coeff, 'degen_mapping', None))
mc0 = mcscf.CASSCF (mf, 5, 2)
mc0.kernel ()

print (getattr (mc0.mo_coeff, 'degen_mapping', None))
print (getattr (mc0.fcisolver.orbsym, 'degen_mapping', None))
mc1 = mcscf.CASCI (mf, 5, 2)
mc1.kernel (mc0.mo_coeff, ci0=mc0.ci)




