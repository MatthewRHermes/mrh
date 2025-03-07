# gpu4mrh is a plugin to use NVIDIA/Intel GPUs in PySCF/MRH package
import functools

def patch_cpu_kernel(cpu_kernel):
    '''Generate a decorator to patch cpu function to gpu function'''
    def patch(gpu_kernel):
        @functools.wraps(cpu_kernel)
        def hybrid_kernel(method, *args, **kwargs):
            return gpu_kernel(method, *args, **kwargs)
#            if getattr(method, 'device', 'cpu') == 'gpu':
#                return gpu_kernel(method, *args, **kwargs)
#            else:
#                return cpu_kernel(method, *args, **kwargs)
        hybrid_kernel.__package__ = 'gpu4mrh'
        return hybrid_kernel
    return patch
