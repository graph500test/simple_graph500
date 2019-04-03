说明： simple-graph500 仅实现简单的并行单GPU版本BFS算法
包含(WS, Direction Optimization, Scan等)  不包含graph generator, SSSP..

图数据保存在./data里  数据名称为kron_20.mtx 

编译 make all

执行方式 ./all ./data/kron_20.mtx
