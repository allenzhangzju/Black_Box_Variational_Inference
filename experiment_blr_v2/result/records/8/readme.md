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
c=5e6 -> b=1~0.226  
M=10  


# abbvi_cv  
num_S=5  
k=0.4  
w=1  
c=5.5e6 -> b=1~0.377  
M=10  

PS: abbvi_cv的b虽然比abbvi_basic的大，但是更低的b导致不稳定，很容易出现-inf
