/* -*- c++ -*- */

#ifndef DEV_ARRAY_H
#define DEV_ARRAY_H

// I don't think we'd need DevArray1D, but maybe if debugging something...

// Don't bother with memory management (i.e. allocate, free) yet; just wrapping objects for now

template<typename T>
class DevArray1D
{
public:

  int l1;   // extent of dimension 1 for index calculation
  int size; // did we allocate memory?
  T * data; // pointer to data

  inline T & operator()(int i1)
  {
    return data[i1];
  }

  DevArray1D()
  {
    l1 = 0;
    size = 0;
    data = nullptr;
  }

  DevArray1D(T * ptr, int n1)
  {
    l1 = n1;
    size = 0;
    data = ptr;
  }
  
  ~DevArray1D()
  {
  }

};
  
template<typename T>
class DevArray2D
{
public:

  int l1;   // extent of dimension 1 for index calculation
  int l2;   // extent of dimension 2 for index calculation (fastest index)
  int size; // did we allocate memory?
  T * data; // pointer to data

  inline T & operator()(int i1, int i2)
  {
    return data[i1 * l2 + i2];
  }

  DevArray2D()
  {
    l1 = 0;
    l2 = 0;
    size = 0;
    data = nullptr;
  }

  DevArray2D(T * ptr, int n1, int n2)
  {
    l1 = n1;
    l2 = n2;
    size = 0;
    data = ptr;
  }
  
  ~DevArray2D()
  {
  }

};
  
template<typename T>
class DevArray3D
{
public:

  int l1;   // extent of dimension 1 for index calculation
  int l2;   // extent of dimension 2 for index calculation
  int l3;   // extent of dimension 3 for index calculation (fastest index)
  int size; // did we allocate memory?
  T * data; // pointer to data

  inline T & operator()(int i1, int i2, int i3)
  {
    return data[(i1 * l2 + i2) * l3 + i3];
  }

  DevArray3D()
  {
    l1 = 0;
    l2 = 0;
    l3 = 0;
    size = 0;
    data = nullptr;
  }

  DevArray3D(T * ptr, int n1, int n2, int n3)
  {
    l1 = n1;
    l2 = n2;
    l3 = n3;
    size = 0;
    data = ptr;
  }
  
  ~DevArray3D()
  {
  }

};
  

#endif
