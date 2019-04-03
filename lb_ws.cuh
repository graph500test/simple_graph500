#ifndef __LB_WS_CUH
#define __LB_WS_CUH

#include "utils.h"
#include "fileIO.h"
#include "scan.cuh"

#include "intrinsics.cuh"

using mask_t=int;
inline void 
reminder(){std::cout << "USING Load-Balanced Strategy: WARP_SEG" << std::endl;}

struct queue_t{
  int64_t build(index_t nvertexs, int CTA_NUM, int THD_NUM){
    int64_t gpu_bytes=0;
    H_ERR(cudaMalloc((void**)&dg_queue, sizeof(index_t)*nvertexs));
    build_tex<int>(dt_queue, dg_queue, nvertexs);

    H_ERR(cudaMalloc((void**)&dg_bin, sizeof(index_t)*BIN_SZ*CTA_NUM*THD_NUM));
    H_ERR(cudaMalloc((void**)&dg_size, sizeof(index_t)*CTA_NUM*THD_NUM));
    H_ERR(cudaMalloc((void**)&dg_offset, sizeof(index_t)*CTA_NUM*THD_NUM));

    H_ERR(cudaMalloc((void**)&dg_qsize, sizeof(index_t)));
    H_ERR(cudaMalloc((void**)&dg_nume, sizeof(index_t)));

    gpu_bytes += sizeof(index_t)*nvertexs + sizeof(index_t);
    gpu_bytes += sizeof(index_t)*CTA_NUM*THD_NUM*(BIN_SZ+2);
    return  gpu_bytes;
  }

  __device__ __forceinline__ 
  index_t get_qsize(){return *dg_qsize;}

  __device__ __forceinline__
  void set_qsize(index_t qs){ *dg_qsize=qs;}

  index_t get_assize(){
    int size;
    cudaMemcpy(&size, dg_qsize, sizeof(index_t), D2H);
    return size;
  }

  int get_nume(){
    int size;
    cudaMemcpy(&size, dg_nume, sizeof(index_t), D2H);
    return size;
  }

  index_t* dg_bin;     //BIN_SZ*CTA_NUM*THD_NUM
  index_t* dg_size;    //CTA_NUM*THD_NUM
  index_t* dg_offset;  //CTA_NUM*THD_NUM
  index_t* dg_queue;   //nvertexs
  index_t* dg_qsize;   //1
  cudaTextureObject_t dt_queue;
  index_t* dg_nume;
};

struct bitmap_t{
  int64_t build(index_t nvertexs){
    int64_t gpu_bytes=0;
    bitmap_size = nvertexs;
    H_ERR(cudaMalloc((void**)&dg_vmask, sizeof(mask_t)*nvertexs));
    H_ERR(cudaMemset((void*)dg_vmask, -1, sizeof(mask_t)*nvertexs));
    return gpu_bytes;
  }

  void set_init_active(int root){
    cudaMemset(dg_vmask+root, 0, sizeof(mask_t));
  }

  // in a read-only way;
  __device__ __forceinline__
  mask_t get_state(const int vid){
    return dg_vmask[vid];
  }

  __device__ __forceinline__
  void mark(const int vid, int lvl){
    dg_vmask[vid] = lvl;
  }

  mask_t* get_mask(){
    mask_t* ret = (mask_t*)malloc(sizeof(mask_t)*bitmap_size);
    cudaMemcpy(ret, dg_vmask, sizeof(mask_t)*bitmap_size, D2H);
    return ret;
  }

  mask_t* dg_vmask;
  index_t bitmap_size;
};

struct active_set_t{
  int64_t build(index_t nvertexs, int CTA_NUM, int THD_NUM){
    int64_t gpu_bytes = queue.build(nvertexs, CTA_NUM, THD_NUM); 
    gpu_bytes += bitmap.build(nvertexs);
    return gpu_bytes;
  }
  queue_t queue;
  bitmap_t bitmap;
};

__global__ void 
__vcull_local_interleave(active_set_t as, graph_t g, int lvl){

  const int STRIDE = blockDim.x*gridDim.x;
  const int gtid   = threadIdx.x + blockIdx.x*blockDim.x;
  const int OFFSET = gtid*BIN_SZ;

  if(gtid == 0) as.queue.dg_nume[0] = 0;
  __syncthreads();
  
  index_t qsize = 0;
  index_t cur_status = 0;
  int nume = 0;

  for(int idx=gtid; idx<g.nvertexs; idx+=STRIDE){
    cur_status  = as.bitmap.get_state(idx);

    if(cur_status == lvl){
      as.queue.dg_bin[OFFSET+qsize] = idx;
	  nume += g.get_out_degree(idx);
      qsize++;
    }
  }
  as.queue.dg_size[gtid] = qsize;
  atomicAdd(as.queue.dg_nume, nume);
}
//pull mode
__global__ void 
__vpull(active_set_t as, graph_t g, int lvl){

  const int STRIDE = blockDim.x*gridDim.x;
  const int gtid   = threadIdx.x + blockIdx.x*blockDim.x;

  const index_t assize = g.nvertexs;
  const index_t* __restrict__ strict_adj_list = g.dg_adj_list;
  const index_t* __restrict__ strict_start_pos = g.dg_start_pos;
  const index_t* __restrict__ strict_odegree = g.dg_odegree;
  
  index_t cur_status = 0;

  for(int idx=gtid; idx<assize; idx+=STRIDE){
    cur_status  = as.bitmap.get_state(idx);
    if(cur_status == -1){
      for(int i=0; i<strict_odegree[idx]; i++){ 
        int ei = strict_start_pos[idx] + i;
        index_t u = strict_adj_list[ei];
	      index_t nei_status = as.bitmap.get_state(u);
	      if(nei_status != -1 && ((nei_status+1)==lvl)){
	        as.bitmap.mark(idx,nei_status+1);
	        break;
        }
      }
    }
  }
}

// push mode
__global__ void 
__vexpand(active_set_t as, graph_t g, int lvl){
  const index_t* __restrict__ strict_adj_list = g.dg_adj_list;

  __shared__ index_t tmp[3*THDNUM];

  const index_t assize = as.queue.get_qsize();
  const int STRIDE  = blockDim.x*gridDim.x;
  const int gtid    = threadIdx.x + blockIdx.x*blockDim.x;
  const int cosize  = 32;
  const int phase   = gtid & (cosize-1);
  const int warp_id = threadIdx.x >> 5;
  const int OFFSET_warp      = 3*cosize*warp_id;
  const int OFFSET_start_pos = OFFSET_warp + cosize;
  const int OFFSET_odegree   = OFFSET_warp + 2*cosize;
  const int assize_align     = (assize&(cosize-1))?(((assize>>5)+1)<<5):assize;

  for(int idx=gtid; idx<assize_align; idx+=STRIDE){
    // step 1: load vertexs into share memory;
    if(idx < assize){
      int v = as.queue.dg_queue[idx];
      tmp[OFFSET_warp+phase] = v;
      tmp[OFFSET_start_pos+phase] = tex1Dfetch<int>(g.dt_start_pos, v);
      tmp[OFFSET_odegree+phase]   = tex1Dfetch<int>(g.dt_odegree, v);
    }else{
      tmp[OFFSET_warp+phase] = -1;
      tmp[OFFSET_odegree+phase] = 0;
    }
    //step 2: get sum of edges for these 32 vertexs and scan odegree;
    int nedges_warp=0;
    int offset=1;
    for(int d=32>>1; d>0; d>>=1){
      if(phase<d){
        int ai = offset*(2*phase+1)-1;
        int bi = offset*(2*phase+2)-1;
        tmp[OFFSET_odegree+bi] += tmp[OFFSET_odegree+ai];
      }
      offset<<=1;
    }
    nedges_warp = tmp[OFFSET_odegree+32-1];
    if(!phase) tmp[OFFSET_odegree+32-1]=0;

    for(int d=1; d<32; d<<=1){
      offset >>=1;
      if(phase<d){
        int ai = offset*(2*phase+1)-1;
        int bi = offset*(2*phase+2)-1;
        int t = tmp[OFFSET_odegree + ai];
        tmp[OFFSET_odegree+ai]  = tmp[OFFSET_odegree+bi];
        tmp[OFFSET_odegree+bi] += t;
      }
    }

    int full_tier = assize_align-cosize;
    int width = idx<(full_tier)?cosize:(assize-full_tier);
    //step 3: process 32 edges in parallel
    for(int i=phase; i<nedges_warp; i+=cosize){
      int id = __upper_bound(&tmp[OFFSET_odegree], width, i)-1;
      if(tmp[OFFSET_warp+id] < 0) continue;
      int ei = tmp[OFFSET_start_pos+id] + i-tmp[OFFSET_odegree+id];
      index_t u = __ldg(strict_adj_list+ei);
      if(as.bitmap.get_state(u) != -1) continue;
      as.bitmap.mark(u, lvl); //lite
    }
  }
}

active_set_t build_active_set(index_t nvertexs){
    active_set_t as;
    std::cout << " -- Allocating memory for Active set storage..." << std::endl;
    int64_t gpu_bytes = as.build(nvertexs, CTANUM, THDNUM);
    std::cout << " -- GPU Global Memory used: " << (0.0+gpu_bytes)/(1ll<<30) << " GB." << std::endl;
    return as;
}

__global__ void 
__compress (const index_t* __restrict__ dg_src,
            index_t* dg_size,
            index_t* dg_offset, index_t* dg_dst, 
            index_t* dg_qsize) {
  const int tid = threadIdx.x + blockIdx.x*blockDim.x;    
  const int n = dg_size[tid];
  const int dst_pos = dg_offset[tid];
  const int src_pos = BIN_SZ*tid;
  for(int i = 0; i < n; i++){
    dg_dst[dst_pos+i] = __ldg(dg_src+src_pos+i);
  }

  if(tid == blockDim.x*gridDim.x-1)
    *dg_qsize = dst_pos+n;
}


__host__ int cull(active_set_t as, graph_t g, int lvl){
  const index_t cta_num = CTANUM>>1;
  const index_t thd_num = THDNUM>>1;
  __vcull_local_interleave<<<CTANUM,THDNUM>>>(as, g, lvl);
  scan<cta_num,thd_num,index_t>(as.queue.dg_size, as.queue.dg_offset, CTANUM*THDNUM);
  __compress<<<CTANUM,THDNUM>>>
    (as.queue.dg_bin, as.queue.dg_size, as.queue.dg_offset, as.queue.dg_queue, as.queue.dg_qsize);
  cudaThreadSynchronize();
  int nactives = as.queue.get_assize();
  return nactives;
}

__host__ void pull(active_set_t as, graph_t g, int lvl){
  __vpull<<<CTANUM,THDNUM>>>(as,g,lvl);
  cudaThreadSynchronize();
}
__host__ void expand(active_set_t as, graph_t g, int lvl){
  __vexpand<<<CTANUM,THDNUM>>>(as,g,lvl);
  cudaThreadSynchronize();
}

#endif
