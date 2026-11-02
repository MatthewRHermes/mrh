# A mini app to do matrix multiplication on cpu
# This is largely used now to benchmark gemms from Python environment compared to directly using C/C++/Fortran libraries (e.g. Intel's MKL)

import numpy as np
import time

n = 1024
num_iter = 1000

a = np.ones((n,n))
b = np.ones((n,n))

t0 = time.time()

for i in range(num_iter):
    np.matmul(a,b)

t1 = time.time() - t0
t1 = t1 / num_iter * 1000.0
print("N= ", n, " time= ", t1, " milliseconds")
