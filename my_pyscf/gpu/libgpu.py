# Hi, I'm a STUB. When a libgpu.so file is present, that will take priority.

# Error messages like the following indicate trying to use the libgpu library, but not having installed/copied it to this directory.
# AttributeError: module 'mrh.my_pyscf.gpu.libgpu' has no attribute 'libgpu_create_device'

def libgpu_create_device():
    raise RuntimeError("ERROR: You're attempting to use the libgpu library, but haven't correctly installed/copied it to mrh/my_pyscf/gpu/libgpu.so.")

def libgpu_init():
    raise RuntimeError("ERROR: You're attempting to use the libgpu library, but haven't correctly installed/copied it to mrh/my_pyscf/gpu/libgpu.so.")
