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

void libgpu_compute_df_get_jk(void * ptr, py::array_t<double> eri1)
{
  py::buffer_info info_eri1 = eri1.request();

  double * ptr_eri1 = static_cast<double*>(info_eri1.ptr);
  
  Device * dev = (Device *) ptr;
  //dev->compute_df_get_jk(data);

  printf("LIBGPU: Inside libgpu_compute_df_get_jk()\n");
  printf("  -- eri1: ndim= %i\n",info_eri1.ndim);
  printf("  --       shape=");
  for(int i=0; i<info_eri1.ndim; ++i) printf(" %i", info_eri1.shape[i]);
  printf("\n");
  printf("  -- eri1[0][0:4]= %f %f %f %f\n", ptr_eri1[0], ptr_eri1[1], ptr_eri1[2], ptr_eri1[3]);
  int off = info_eri1.shape[1];
  printf("  -- eri1[1][0:4]= %f %f %f %f\n", ptr_eri1[off+0], ptr_eri1[off+1], ptr_eri1[off+2], ptr_eri1[off+3]);
  
}

/* ---------------------------------------------------------------------- */

void libgpu_orbital_response(void * ptr, int nmo, py::array_t<double> ppaa)
{
  printf("HELLO from libgpu_orbital_respons()!!\n");
  
  py::buffer_info info_ppaa = ppaa.request();

  double * ptr_ppaa = static_cast<double*>(info_ppaa.ptr);
  
  printf("LIBGPU: Inside libgpu_orbital_response()\n");
  printf("  -- nmo= %i\n",nmo);
  printf("  -- ppaa: ndim= %i\n",info_ppaa.ndim);
  printf("  --       shape=");
  for(int i=0; i<info_ppaa.ndim; ++i) printf(" %i", info_ppaa.shape[i]);
  printf("\n");
}
