# The historical recordes for compare  
## records 1, 2, 3, 4  
batchSize=120, num_S=10, epoch=15, num_St=5000,  
固定abbvi_basic的参数，测试bbvi不同步长下的性能
## records 5  
batchSiza=120, num_S=5, epoch=15, num_St=2000（为了跑快点，应该没啥影响）  
降低num_S，测试两种算法的性能  
注：abbvi_basic的c需要重新调整，b太小了算法不稳定，  
records 5 的b比records 1,2,3,4 的要大些  
## records 6  
batchSize=64, num_S=5,epoch=15, num_St=2000
降低了batchSize，两种算法性能间的差异更小，且波动大
