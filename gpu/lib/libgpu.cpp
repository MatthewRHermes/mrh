#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "libgpu.h"
#include "pm.h"

// Fortran
//  :: allocate(N, M)
//  :: access (i,j)
// C/C++
//  :: access[(j-1)*N + (i-1)]

#define F_2D(name, i, j, lda) name[(j-1)*lda + i-1]

// might have order of lds flipped...
#define F_3D(name, i, j, k, lda, ldb) name[(k-1)*lda*ldb + (j-1)*lda + i-1]
#define F_4D(name, i, j, k, l, lda, ldb, ldc) name[(l-1)*lda*ldb*ldc + (k-1)*lda*ldb + (j-1)*lda + i-1]

void * libgpu_create_device()
{
  Device * ptr = new Device();
  return (void *) ptr;
}

/* ---------------------------------------------------------------------- */

void libgpu_destroy_device(void * ptr)
{
  Device * dev = (Device *) ptr;
  delete dev;
}

/* ---------------------------------------------------------------------- */

int libgpu_get_num_devices(void * ptr)
{
  Device * dev = (Device *) ptr;
  return dev->get_num_devices();
}

/* ---------------------------------------------------------------------- */

void libgpu_dev_properties(void * ptr, int N)
{
  Device * dev = (Device *) ptr;
  dev->get_dev_properties(N);
}

/* ---------------------------------------------------------------------- */

void libgpu_set_device(void * ptr, int id)
{
  Device * dev = (Device *) ptr;
  dev->set_device(id);
}

/* ---------------------------------------------------------------------- */

void libgpu_setup(void * ptr, py::array_t<double> array, int N)
{
  py::buffer_info info = array.request();
  auto data = static_cast<double*>(info.ptr);
  
  Device * dev = (Device *) ptr;
  dev->setup(data,N);
}

/* ---------------------------------------------------------------------- */

double libgpu_compute(void * ptr, py::array_t<double> array)
{
  py::buffer_info info = array.request();
  auto data = static_cast<double*>(info.ptr);
  
  Device * dev = (Device *) ptr;
  return dev->compute(data);
}

/* ---------------------------------------------------------------------- */
