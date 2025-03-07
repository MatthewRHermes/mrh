import numpy as np
import baseline
import libgpu
gpu = libgpu.libgpu_create_device()

num_gpus = libgpu.libgpu_get_num_devices(gpu)
print("num_gpus= ", num_gpus)

libgpu.libgpu_dev_properties(gpu, num_gpus)

gpu_id = 0
libgpu.libgpu_set_device(gpu, gpu_id)

# ------------------------------------------- #

ncore, nocc, nmo = (7, 9, 26) # nfrags = 1
#ncore, nocc, nmo = (13, 17, 48) # nfrags = 2
#ncore, nocc, nmo = (25, 33, 92) # nfrags = 4
#ncore, nocc, nmo = (49, 65, 180) # nfrags = 8
#ncore, nocc, nmo = (97, 129, 356) # nfrags = 16
#ncore, nocc, nmo = (193, 257, 708) # nfrags = 32
#ncore, nocc, nmo = (385, 513, 1412) # nfrags = 64

gap = nocc - ncore

size = nmo * nmo * gap * gap
size += nmo * gap * nmo * gap
size += nmo * gap * gap * gap

size += gap * gap * gap * nmo
size += gap * gap * gap * gap
size += nmo * nmo

size *= 8 / (1024 * 1024 * 1024)

print("Allocating ", size, "GBs")

ppaa = np.random.rand(nmo, nmo, gap, gap)
papa = np.random.rand(nmo, gap, nmo, gap)
paaa = np.random.rand(nmo, gap, gap, gap)

ocm2_orig = np.random.rand(gap, gap, gap, nmo)
tcm2_orig = np.random.rand(gap, gap, gap, gap)
gorb = np.random.rand(nmo, nmo)

ocm2 = ocm2_orig
tcm2 = tcm2_orig
ref = baseline.orbital_response(ppaa, papa, paaa, ocm2, tcm2, gorb, ncore, nocc, nmo)

f1_prime = np.zeros((nmo, nmo), dtype=np.float64)
ocm2 = ocm2_orig
tcm2 = tcm2_orig
libgpu.libgpu_orbital_response(gpu, f1_prime, ppaa, papa, paaa, ocm2, tcm2, gorb, ncore, nocc, nmo)

#print("\nref(", ref.shape, ")= ", ref[0])
#print("\nf1_prime(", f1_prime.shape, ")= ", f1_prime[0])

#for i in range(nmo):
#    err = 0.0
#    for j in range(nmo):
#        err += (ref[i,j] - f1_prime[i,j]) * (ref[i,j] - f1_prime[i,j])
#    print("i= ", i, "  err= ", err)

err = 0.0
for i in range(nmo):
    for j in range(nmo):
        err += (ref[i,j] - f1_prime[i,j]) * (ref[i,j] - f1_prime[i,j])

print("\nError= ", err)

num_iter = 10
for i in range(num_iter):
    ocm2 = ocm2_orig
    tcm2 = tcm2_orig
    ref = baseline.orbital_response(ppaa, papa, paaa, ocm2, tcm2, gorb, ncore, nocc, nmo)

for i in range(num_iter):
    f1_prime = np.zeros((nmo, nmo), dtype=np.float64)
    ocm2 = ocm2_orig
    tcm2 = tcm2_orig
    libgpu.libgpu_orbital_response(gpu, f1_prime, ppaa, papa, paaa, ocm2, tcm2, gorb, ncore, nocc, nmo)

# ------------------------------------------- #

libgpu.libgpu_destroy_device(gpu)
