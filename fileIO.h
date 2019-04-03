#ifndef __FILEIO_H
#define __FILEIO_H

#include <type_traits>
#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <string>

#include "utils.h"

template<typename data_t>
void build_tex(cudaTextureObject_t &tex_obj, data_t* buf, int N){
  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeLinear;
  resDesc.res.linear.devPtr = buf;
  if(std::is_same<int,data_t>::value) resDesc.res.linear.desc.f = cudaChannelFormatKindSigned;
  else if(std::is_same<float, data_t>::value) resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
  else ASSERT(false, "build texture w/ bad data type");
  resDesc.res.linear.desc.x = 32;
  resDesc.res.linear.sizeInBytes = N*sizeof(data_t);

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = cudaReadModeElementType;

  H_ERR(cudaCreateTextureObject(&tex_obj, &resDesc, &texDesc, NULL));
}

inline off_t fsize(const char *filename) {
    struct stat st; 
    if (stat(filename, &st) == 0)
        return st.st_size;
    return -1; 
}

template<typename data_t>
struct rand_device{};

template<>
struct rand_device<float>{
  static inline float rand_weight(int lim){
    return (float)drand48()*lim;
  }
};

template<>
struct rand_device<int>{
  static inline int rand_weight(int lim){
    return (int)(1+rand()%lim);
  }
};

// normailze index start from 0
class edgelist_t{
public:
  void read_mtx(std::string path, 
                bool with_header=false){
    double start = wtime();

    std::ifstream fin(path);
    if(!fin.is_open()) ASSERT(false, "can not open file");

    // skip comments
    while(1){
      char c = fin.peek();
      if(c>='0' && c<='9') break;
      fin.ignore(std::numeric_limits<std::streamsize>::max(), fin.widen('\n'));
    }

    // if with header (meaningless, just ignore)
    if(with_header) fin >> nvertexs >> nvertexs >> nedges;
    else nvertexs = nedges = -1;

    vvector.clear();

    vmin=std::numeric_limits<index_t>::max();
    vmax=-1;
    while(fin.good()){
      index_t v0, v1;
      fin >> v0 >> v1;
      fin.ignore(std::numeric_limits<std::streamsize>::max(), fin.widen('\n'));
      if(fin.eof()) break;
      if(v0 == v1) continue;

      vmin = std::min(v0,vmin);
      vmin = std::min(v1,vmin);
      vmax = std::max(v0,vmax);
      vmax = std::max(v1,vmax);

      if(fin.eof()) break;

      vvector.push_back(v0);
      vvector.push_back(v1);
    }

    nvertexs = (vmax-vmin+1);
    nedges = vvector.size()>>1;

    fin.close();
    double end = wtime();
    std::cout << "IO time: " << end-start << " s." << std::endl;
  }

public:
  std::vector<index_t> vvector;
  index_t vmin, vmax;
  index_t nvertexs;
  index_t nedges;
};


class host_graph_t{
public:
  void build(edgelist_t el){
    double start = wtime();

    int64_t vmax = el.vmax;
    int64_t vmin = el.vmin;

    int64_t mem_used = 0;
    nvertexs = (vmax-vmin+1);
    nedges = el.vvector.size()>>1;

    nedges <<= 1;

    odegrees = (index_t*)calloc(nvertexs, sizeof(index_t));
    start_pos = (index_t*)malloc(sizeof(index_t)*nvertexs);
    mem_used += sizeof(index_t)*nvertexs*2;


    for(size_t i = 0; i < el.vvector.size(); i+=2) {
      odegrees[el.vvector[i]-vmin] ++;
      odegrees[el.vvector[i+1]-vmin] ++;
    }

    start_pos[0] = 0;
    for(index_t i = 1; i < nvertexs; i++){
      start_pos[i] = odegrees[i-1] + start_pos[i-1];
      odegrees[i-1] = 0;
    }
    odegrees[nvertexs-1]=0;
    

    adj_list = (index_t*)malloc(sizeof(index_t)*nedges);
    mem_used += sizeof(index_t)*nedges;

    for(size_t i = 0; i < (el.vvector.size()>>1); i++){
      index_t v0 = el.vvector[i<<1] - vmin;
      index_t v1 = el.vvector[i<<1|1] - vmin;

      adj_list[start_pos[v0] + odegrees[v0]] = v1;
      odegrees[v0] ++;

      adj_list[start_pos[v1] + odegrees[v1]] = v0;
      odegrees[v1] ++;
    }

    index_t min_odgreee, max_odegree;
    mean_odegree = min_odgreee = max_odegree = odegrees[0];
    for(index_t i = 1; i < nvertexs; i++){
      min_odgreee = std::min(odegrees[i], min_odgreee);
      max_odegree = std::max(odegrees[i], max_odegree);
      mean_odegree += odegrees[i];
    }
    mean_odegree = (0.0+mean_odegree)/nvertexs;

    double end = wtime();
    std::cout << "CSR transform time: " << end-start << " s." << std::endl;
    std::cout << " -- nvertexs: " << nvertexs << " nedges: " << nedges << std::endl;
    std::cout << " -- degree (min, mean, max): (" << min_odgreee << ", " << mean_odegree << ", "<< max_odegree << ")" << std::endl;
    std::cout << "Host Graph memory used: " << (0.0+mem_used+2*sizeof(index_t)+2*sizeof(bool))/(1l<<30) << " Gb."<< std::endl;
  }

 public:
  index_t* adj_list;
  index_t* start_pos;
  index_t* odegrees;
  index_t nvertexs;
  index_t nedges;
  double mean_odegree;
};

// vertex-centric
struct graph_t{

  int64_t build(host_graph_t hgraph){
    this->nvertexs = hgraph.nvertexs;
    this->nedges   = hgraph.nedges;
    this->level    = 0;

    int64_t gpu_bytes=0;
    H_ERR(cudaMalloc((void**)&dg_adj_list,  sizeof(index_t)*hgraph.nedges)); 
    H_ERR(cudaMalloc((void**)&dg_odegree,   sizeof(index_t)*hgraph.nvertexs));
    H_ERR(cudaMalloc((void**)&dg_start_pos, sizeof(index_t)*hgraph.nvertexs));

    H_ERR(cudaMemcpy(dg_adj_list,  hgraph.adj_list,  sizeof(index_t)*hgraph.nedges, H2D));
    H_ERR(cudaMemcpy(dg_odegree,   hgraph.odegrees,  sizeof(index_t)*hgraph.nvertexs, H2D));
    H_ERR(cudaMemcpy(dg_start_pos, hgraph.start_pos, sizeof(index_t)*hgraph.nvertexs, H2D));

    build_tex<int>(dt_odegree,   dg_odegree,   nvertexs);
    build_tex<int>(dt_start_pos, dg_start_pos, nvertexs);

    gpu_bytes += sizeof(index_t)*nvertexs*3;
    gpu_bytes += sizeof(index_t)*nedges;

    return gpu_bytes;
  }

  __device__ __forceinline__
  index_t get_out_degree(const int vid){
    return tex1Dfetch<int>(dt_odegree, vid); 
  }

  __device__ __forceinline__
  index_t get_level(){
    return level;
  }

  //WARNING,TODO: This update is only vaild in the top level
  // since the primitives we have accept parameter by assignment
  inline void update_level(int inc = 1){
    level+=inc;
  }

  index_t* dg_adj_list;
  index_t* dg_odegree;
  index_t* dg_start_pos;
  index_t nvertexs;
  index_t nedges;
  index_t level;
  cudaTextureObject_t dt_odegree;
  cudaTextureObject_t dt_start_pos;
};

graph_t build_graph(host_graph_t hgraph){
  graph_t g;
  int gpu_bytes=0;
  std::cout << " -- Allocating memory for graph storage..." << std::endl;
  gpu_bytes += g.build(hgraph);
  std::cout << " -- GPU Global Memory used: " << (0.0+gpu_bytes)/(1ll<<30) << " GB." << std::endl;
  return g;
}



#endif
