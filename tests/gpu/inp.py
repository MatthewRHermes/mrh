import numpy as np
import cpu_inp
import gpu_inp
nfrags = 4
basis = 'sto-3g'

cpu_mf, cpu_las = cpu_inp.run_cpu(nfrags,basis)
gpu_mf, gpu_las = gpu_inp.run_gpu(nfrags,basis)
assert np.abs(cpu_mf.e_tot - gpu_mf.e_tot) < 1e-6
assert np.abs(cpu_las.e_tot - gpu_las.e_tot) < 1e-6
