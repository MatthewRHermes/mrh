#if defined(_USE_GPU)

#if defined(_GPU_CUDA)
#include "pm_cuda.h"
#endif

#elif defined(_USE_CPU)

#include "pm_host.h"

#endif
