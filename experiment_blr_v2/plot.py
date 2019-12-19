import numpy as np
import matplotlib.pyplot as plt

bbvi_basic=np.load('./result/bbvi_basic.npy')
abbvi_basic=np.load('./result/abbvi_basic.npy')

x=np.array(range(len(bbvi_basic)))


plt.plot(x,bbvi_basic,label='bbvi_basic',alpha=0.8)
plt.plot(x,abbvi_basic,label='abbvi_basic',alpha=0.8)
plt.legend()
plt.grid()
plt.show()


a=1