#include <iostream>
#include <queue>
#include "utils.h"
#include "lb_ws.cuh"
#include "fileIO.h"

using G = graph_t;

int* bfs_cpu(host_graph_t hg, int root){
  std::cout << "generate CPU BFS reference" << std::endl;
  double ms = wtime();
  int* lvl = (int*)malloc(sizeof(int)*hg.nvertexs);
  memset(lvl,-1, sizeof(int)*hg.nvertexs);
  std::queue<int> q;
  lvl[root] = 0;
  q.push(root);
  while(!q.empty()){
    int v = q.front();
    q.pop();
    int s = hg.start_pos[v];
    int e = (v==(hg.nvertexs-1)?hg.nedges:hg.start_pos[v+1]);
    for(int j=s; j<e; ++j){
      int u = hg.adj_list[j];
      if (lvl[u]==-1){
        lvl[u] = lvl[v] + 1;
        q.push(u);
      }
    }
  }
  double me = wtime();
  std::cout << "CPU BFS: " << (me-ms)*1000 << " ms" << std::endl;
  return lvl;
}

void validation(int* lCPU, mask_t* lGPU, int N){
  bool flag=true;
  for(int i=0;i<N;++i){
    if(lGPU[i]-lCPU[i] != 0){
      flag = false;
      puts("failed");
      printf("%d %d %d\n",i,lGPU[i],lCPU[i]);
      break;
    }
  }
  if(flag) puts("passed");
}


double run_bfs(graph_t g, mask_t** vdata, int root){
  active_set_t as = build_active_set(g.nvertexs);

  // step 1: initializing
  std::cout << " -- Initializing ..." << std::endl;
  as.bitmap.set_init_active(root);
	
  double s = wtime();
  double ms=s,me=s;
  // step 2: Execute Algorithm
  for(int level=0;;level++){
    ms = wtime();
    index_t nactives = cull(as,g,level);
    index_t naedges = as.queue.get_nume();
    me = wtime();
    double tt = me-ms;

    if(nactives==0) break;

    ms = wtime();
    //Direction switching condition can be more complicate.
    //Simple Version: first 2 levels:top-down, other levels:bottom-up
    if(level < 2){
      expand(as,g,level+1);
    }else{
      pull(as,g,level+1);
    }
    me = wtime();

    printf("Level %d (%.4f %.4f ms): %d\n",level,tt*1000, (me-ms)*1000, nactives);
  }
  double e = wtime();

  *vdata=as.bitmap.get_mask();
  // step 3: write back result
  return e-s;
}

int main(int argc, char* argv[]){
  set_device(2);
  edgelist_t el;
  if(argc < 2) return -1;
  el.read_mtx(argv[1]);

  host_graph_t hg;
  hg.build(el);

  // step 1 : choose root vertex
  int root=0;
  std::cout << " -- BFS root is: " << root << std::endl;

  // step 2 : init Algorithm
  graph_t g = build_graph(hg);

  std::cout << " -- Launching BFS" << std::endl;
  reminder();
  // step 3 : execute Algorithm
  mask_t* vdata=NULL;
  double time = run_bfs(g,&vdata,root);
    
  // step 4 : validation
  int* lvl = bfs_cpu(hg, root);
  validation(lvl, vdata, hg.nvertexs);

  std::cout << "BFS time: "<< 1000*time << " ms." << std::endl;

  return 0;
}
