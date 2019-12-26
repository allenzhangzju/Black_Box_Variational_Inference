# setting:  
## both  
num_epochs=15  
batchSize=120  
num_St=2000  

  
## bbvi & bbvi_cv  
num_S=5#训练的采样数量  
eta=0.3
  

# abbvi  
num_S=5#训练的采样数量  
eta=0.05  
k=0.4  
w=1  
c=10e6 -> b=1~0.226  
M=10  


# abbvi_cv  
num_S=5  
k=0.4  
w=1  
c=5.5e6 -> b=1~0.38  
M=10  

PS: abbvi_cv的b虽然不小，但是已经很极限
