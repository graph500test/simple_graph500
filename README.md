Note： simple-graph500 is only a simple parallel BFS version on single GPU.
      Including(WS, Direction Optimization, Scan...)  
      No including graph generator, SSSP..

Dataset：In current dir, including test.data (32 vertices，128 edges)
     If need more dataset，download at http://networkrepository.com/graph500.php 

Evironment： g++4.8 cuda7.5 -O3

Complier： make all

Run： ./all ./test.data 
