import numpy as np
import matplotlib.pyplot as plt

bbvi_basic=np.load('./result/records/7/bbvi_basic.npy')
abbvi_basic=np.load('./result/records/7/abbvi_basic.npy')

x=np.array(range(len(bbvi_basic)))


plt.plot(x,bbvi_basic,label='bbvi_basic',alpha=0.8)
plt.plot(x,abbvi_basic,label='abbvi_basic',alpha=0.8)
plt.legend()
plt.grid()
plt.show()

'''
setting:
num_epochs=30
batchSize=240

num_S=5#训练的采样数量
eta=0.3

num_S=5#训练的采样数量
eta=0.05
k=0.4
w=1
c=10e6 -> b=1~0.226
M=10
'''
