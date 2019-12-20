import numpy as np
import matplotlib.pyplot as plt

bbvi_null=np.load('./result/bbvi_null.npy')
bbvi_rao_blackwellization=np.load('./result/bbvi_rao_blackwellization.npy')
bbvi_cv=np.load('./result/bbvi_cv.npy')
bbvi_repara=np.load('./result/bbvi_repara.npy')
bbvi_rbcv=np.load('./result/bbvi_rbcv.npy')

x=np.array(range(len(bbvi_null[0])))



plt.plot(x,bbvi_rao_blackwellization[0],label='rao_blackwell')
plt.plot(x,bbvi_cv[0],label='cv')
plt.plot(x,bbvi_rbcv[0],label='rb+cv')
plt.plot(x,bbvi_repara[0],label='repara')
plt.plot(x,bbvi_null[0],label='basic')
plt.legend()
plt.figure()
plt.yscale('log')
plt.plot(x,bbvi_null[1],label='basic')
plt.plot(x,bbvi_rao_blackwellization[1],label='rao_blackwell')
plt.plot(x,bbvi_cv[1],label='cv')
plt.plot(x,bbvi_rbcv[1],label='rb+cv')
plt.plot(x,bbvi_repara[1],label='repara')
plt.legend()
plt.show()


a=1