#ifndef __INTRINSICS_CUH
#define __INTRINSICS_CUH


__device__ __forceinline__ int
__lower_bound(int* array, int len, int key){
  int s = 0;
  while(len>0){
    int half = len>>1;
    int mid = s + half;
    if(array[mid] < key){
      s = mid + 1;
      len = len-half-1;
    }else{
      len = half;
    }
  }
  return s;
}

__device__ __forceinline__ int
__upper_bound(int* array, int len, int key){
  int s = 0;
  while(len>0){
    int half = len>>1;
    int mid = s + half;
    if(array[mid] > key){
      len = half;
    }else{
      s = mid+1;
      len = len-half-1;
    }
  }
  return s;
}

/// codes from B40C and Gunrock
#if defined(_WIN64) || defined(__LP64__)
    #define _GS_LP64__ 1
    #define _REG_PTR_ "l"
#else
    #define _GS_LP64__ 0
    #define _REG_PTR_ "r"
#endif

enum CacheHint{DFT,wb,cg,cs,wt};

template <CacheHint CACHEHINT>
struct Proxy{
  template <typename T>
  __device__ __forceinline__ static void St(T val, T *ptr);
};


// cache aware store
#if __CUDA_ARCH__ >= 200
  template <>
  template <typename T>
  __device__ __forceinline__ void 
  Proxy<DFT>::St(T val, T *ptr){*ptr = val;}

  // Singleton store op
  #define STORE(base_type, ptx_type, reg_mod, cast_type, hint)                                                                  \
    template<> template<> void Proxy<hint>::St(base_type val, base_type* ptr) {                                                 \
      asm volatile ("st.global."#hint"."#ptx_type" [%0], %1;" : : _REG_PTR_(ptr), #reg_mod(reinterpret_cast<cast_type&>(val))); \
    }

  // Defines specialized store ops for three cache hinter
  #define STORE_BASE(base_type, ptx_type, reg_mod, cast_type) \
    STORE(base_type, ptx_type, reg_mod, cast_type, cg)        \
    STORE(base_type, ptx_type, reg_mod, cast_type, wb)        \
    STORE(base_type, ptx_type, reg_mod, cast_type, cs)        

#if CUDA_VERSION >= 4000
  #define _REG8         h
  #define _REG16        h
  #define _CAST8        short
#else
  #define _REG8         r
  #define _REG16        r
  #define _CAST8        char
#endif

  STORE_BASE(char,           s8,  _REG8,  _CAST8)
  STORE_BASE(unsigned char,  u8,  _REG8,  unsigned _CAST8)
  //STORE_BASE(signed char,    s8,  r,      signed int)
  STORE_BASE(bool,           s8,  r,      unsigned int)
  STORE_BASE(short,          s16, _REG16, short)
  STORE_BASE(int,            s32, r,      int)
  STORE_BASE(unsigned short, u16, _REG16, unsigned short)
  STORE_BASE(unsigned int,   u32, r,      unsigned int)
  STORE_BASE(float,          f32, f,      float)
#if !defined(__LP64__) || (__LP64__ == 0)
  STORE_BASE(long,           s32, r,      long)
  STORE_BASE(unsigned long,  u32, r,      unsigned long)
#else
  STORE_BASE(long,           s64, l,      long)
  STORE_BASE(unsigned long,  u64, l,      unsigned long)
#endif
  STORE_BASE(unsigned long long, u64, l, unsigned long long)
  STORE_BASE(long long,          s64, l, long long)
  STORE_BASE(double,             s64, l, long long) 

  #undef STORE_BASE
  #undef _CAST8
  #undef _REG8
  #undef _REG16

#else  //__CUDA_ARCH__
  template <CacheHint WRITE_MODIFIER>
  template <typename T>
  __device__ __forceinline__ void 
  Proxy<WRITE_MODIFIER>::St(T val, T *ptr){*ptr = val;}
#endif //__CUDA_ARCH__




#endif
