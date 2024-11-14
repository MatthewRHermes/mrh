
./a.out

#ONEAPI_DEVICE_SELECTOR=level_zero:*.* ./a.out

#nsys profile --stats=true ./a.out

# WARNING :: set _NUM_ITERATIONS_GPU to something small (e.g. 5)
#ncu --print-summary per-kernel ./a.out
