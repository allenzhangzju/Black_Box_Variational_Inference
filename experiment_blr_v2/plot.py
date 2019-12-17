import numpy as np
import matplotlib.pyplot as plt

bbvi_basic=np.load('./result/datas/bbvi_basic.npy')

x=np.array(range(len(bbvi_basic)))


plt.plot(x,bbvi_basic,label='basic')
plt.show()


a=1