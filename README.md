说明： simple-graph500 仅实现简单的并行单GPU版本BFS算法
      包含(WS, Direction Optimization, Scan等)  不包含graph generator, SSSP..

数据：当前目录下有一个测试图test.data 包含32个顶点，128条边
     若需要测试更多的图数据，可以去http://networkrepository.com/graph500.php 下载

环境： g++4.8 cuda7.5 -O3

编译： make all

执行方式： ./all ./test.data 
