import numpy as np
import matplotlib.pyplot as plt

bbvi_null=np.load('./result/bbvi_null.npy')
bbvi_rao_blackwellization=np.load('./result/bbvi_rao_blackwellization.npy')

x=np.array(range(len(bbvi_null[0])))


plt.plot(x,bbvi_null[0],label='basic')
plt.plot(x,bbvi_rao_blackwellization[0],label='rao_blackwell')
plt.legend()
plt.figure()
plt.yscale('log')
plt.plot(x,bbvi_null[1],label='basic')
plt.plot(x,bbvi_rao_blackwellization[1],label='rao_blackwell')
plt.legend()
plt.show()


a=1