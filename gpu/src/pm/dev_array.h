/* -*- c++ -*- */

#ifndef DEV_ARRAY_H
#define DEV_ARRAY_H

#include "pm.h"

using namespace PM_NS;

#define DA_HOST 0
#define DA_DEVICE 1
#define DA_BOTH 2

#define DA_HostToDevice 0
#define DA_DeviceToHost 1

template<typename T>
class DevArray1D
{
public:

  int l1;      // extent of dimension 1 for index calculation
  size_t size; // memory allocated
  T * h_ptr;   // pointer to host data
  T * d_ptr;   // pointer to device data
  
  class PM * pm;

  inline T & operator()(int i1) // we only offer this for the host right now
  {
    return h_ptr[i1];
  }

  DevArray1D(class PM * pm_)
  {
    l1 = 0;
    size = 0;
    h_ptr = nullptr;
    d_ptr = nullptr;
    pm = pm_;
  }

  DevArray1D(T * ptr, int n1, class PM * pm_, int mode = DA_HOST)
  {
    l1 = n1;
    size = n1;
    h_ptr = nullptr;
    d_ptr = nullptr;
    pm = pm_;

    if(mode == DA_HOST) h_ptr = ptr;
    if(mode == DA_DEVICE) d_ptr = ptr;
    if(mode == DA_BOTH) {
      printf("Error :: DevArray1D can't be initialized with DA_BOTH");
      exit(1);
    }
  }
  
  ~DevArray1D()
  {
  }

  void reshape(int n1) // useful?? 
  {
    l1 = n1;
    
    if(l1 > size) {
      printf("Error :: DevArray1D reshaped, but larger than space allocated\n");
      exit(1);
    }
  }
  
  void safe_grow(int n1, int mode = DA_HOST)
  {
    l1 = n1; // always update to handle any reshaping
    
    if(l1 > size) {
      size = l1;

      if(mode == DA_HOST || mode == DA_BOTH) {
	if(h_ptr) pm->dev_free_host(h_ptr);
	h_ptr = (T *) pm->dev_malloc_host(size * sizeof(T));
      }

      if(mode == DA_DEVICE || mode == DA_BOTH) {
	if(d_ptr) pm->dev_free(d_ptr);
	d_ptr = (T *) pm->dev_malloc(size * sizeof(T));
      }
    }
  }

  void free(int mode = DA_HOST)
  {
    if(mode == DA_HOST || mode == DA_BOTH) {
      if(h_ptr) pm->dev_free_host(h_ptr);
    }
    
    if(mode == DA_DEVICE || mode == DA_BOTH) {
      if(d_ptr) pm->dev_free(d_ptr);
    }
  }
  
};
  
template<typename T>
class DevArray2D
{
public:

  int l1;      // extent of dimension 1 for index calculation
  int l2;      // extent of dimension 2 for index calculation (fastest index)
  size_t size; // memory allocated
  T * h_ptr;   // pointer to host data
  T * d_ptr;   // pointer to device data

  class PM * pm;
  
  inline T & operator()(int i1, int i2) // we only offer this for the host right now
  {
    return h_ptr[i1 * l2 + i2];
  }

  DevArray2D(class PM * pm_)
  {
    l1 = 0;
    l2 = 0;
    size = 0;
    h_ptr = nullptr;
    d_ptr = nullptr;
    pm = pm_;
  }

  DevArray2D(T * ptr, int n1, int n2, class PM * pm_, int mode = DA_HOST)
  {
    l1 = n1;
    l2 = n2;
    size = n1 * n2;
    h_ptr = nullptr;
    d_ptr = nullptr;
    pm = pm_;
    
    if(mode == DA_HOST) h_ptr = ptr;
    if(mode == DA_DEVICE) d_ptr = ptr;
    if(mode == DA_BOTH) {
      printf("Error :: DevArray2D can't be initialized with DA_BOTH");
      exit(1);
    }
  }
  
  ~DevArray2D() // no memory should implicitly be removed
  {
  }

  void reshape(int n1, int n2)
  {
    l1 = n1;
    l2 = n2;
    
    if(l1*l2 > size) {
      printf("Error :: DevArray2D reshaped, but larger than space allocated\n");
      exit(1);
    }
  }
  
  void safe_grow(int n1, int n2, int mode = DA_HOST)
  {
    l1 = n1; // always update to handle any reshaping
    l2 = n2;

    if(l1*l2 > size) {
      size = l1*l2;

      if(mode == DA_HOST || mode == DA_BOTH) {
	if(h_ptr) pm->dev_free_host(h_ptr);
	h_ptr = (T *) pm->dev_malloc_host(size * sizeof(T));
      }

      if(mode == DA_DEVICE || mode == DA_BOTH) {
	if(d_ptr) pm->dev_free(d_ptr);
	d_ptr = (T *) pm->dev_malloc(size * sizeof(T));
      }
    }
  }

  void free(int mode = DA_HOST)
  {
    if(mode == DA_HOST || mode == DA_BOTH) {
      if(h_ptr) pm->dev_free_host(h_ptr);
    }
    
    if(mode == DA_DEVICE || mode == DA_BOTH) {
      if(d_ptr) pm->dev_free(d_ptr);
    }
  }
};
  
template<typename T>
class DevArray3D
{
public:

  int l1;      // extent of dimension 1 for index calculation
  int l2;      // extent of dimension 2 for index calculation
  int l3;      // extent of dimension 3 for index calculation (fastest index)
  size_t size; // memory allocated
  T * h_ptr;   // pointer to host data
  T * d_ptr;   // pointer to device data 

  class PM * pm;
  
  inline T & operator()(int i1, int i2, int i3) // we only offer this for the host right now
  {
    return h_ptr[(i1 * l2 + i2) * l3 + i3];
  }

  DevArray3D(class PM * pm_)
  {
    l1 = 0;
    l2 = 0;
    l3 = 0;
    size = 0;
    h_ptr = nullptr;
    d_ptr = nullptr;
    pm = pm_;
  }

  DevArray3D(T * ptr, int n1, int n2, int n3, class PM * pm_, int mode = DA_HOST)
  {
    l1 = n1;
    l2 = n2;
    l3 = n3;
    size = n1 * n2 * n3;
    h_ptr = nullptr;
    d_ptr = nullptr;
    pm = pm_;
    
    if(mode == DA_HOST) h_ptr = ptr;
    if(mode == DA_DEVICE) d_ptr = ptr;
    if(mode == DA_BOTH) {
      printf("Error :: DevArray3D can't be initialized with DA_BOTH\n");
      exit(1);
    }
  }
  
  ~DevArray3D() // no memory should implicitly be removed
  {
  }

  void reshape(int n1, int n2, int n3)
  {
    l1 = n1;
    l2 = n2;
    l3 = n3;

    if(l1*l2*l3 > size) {
      printf("Error :: DevArray3D reshaped, but larger than space allocated\n");
      exit(1);
    }
  }
  
  void safe_grow(int n1, int n2, int n3, int mode = DA_HOST)
  {
    l1 = n1; // always update to handle any reshaping
    l2 = n2;
    l3 = n3;

    if(l1*l2*l3 > size) {
      size = l1*l2*l3;

      if(mode == DA_HOST || mode == DA_BOTH) {
	if(h_ptr) pm->dev_free_host(h_ptr);
	h_ptr = (T *) pm->dev_malloc_host(size * sizeof(T));
      }

      if(mode == DA_DEVICE || mode == DA_BOTH) {
	if(d_ptr) pm->dev_free(d_ptr);
	d_ptr = (T *) pm->dev_malloc(size * sizeof(T));
      }
    }
  }

  void free(int mode = DA_HOST)
  {
    if(mode == DA_HOST || mode == DA_BOTH) {
      if(h_ptr) pm->dev_free_host(h_ptr);
    }
    
    if(mode == DA_DEVICE || mode == DA_BOTH) {
      if(d_ptr) pm->dev_free(d_ptr);
    }
  }
};
#endif
