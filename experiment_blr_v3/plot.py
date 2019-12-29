import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

bbvi_basic=np.load('./result_elbo/bbvi_basic.npy')
bbvi_cv=np.load('./result_elbo/bbvi_cv.npy')
abbvi_basic=np.load('./result_elbo/abbvi_basic.npy')
repara=np.load('./result_elbo/ref_repara.npy')
abbvi_cv=np.load('./result_elbo/abbvi_cv.npy')
#test=np.load('./result_para/abbvi_basic.npy')
'''
abbvi_basic=np.load('./result/abbvi_basic.npy')
bbvi_cv=np.load('./result/bbvi_cv.npy')
abbvi_cv=np.load('./result/abbvi_cv.npy')
ref=np.load('./result/ref_repara.npy')
test=np.load('./result/test.npy')
'''
x=np.array(range(len(bbvi_basic)))


#plt.plot(x,ref,label='repara',alpha=1,color='r')
#plt.plot(x,test,label='test',alpha=0.8)

plt.plot(x,repara,label='repara',alpha=0.8)
plt.plot(x,abbvi_basic,label='abbvi_basic',alpha=0.8)
plt.plot(x,bbvi_cv,label='bbvi_cv',alpha=0.8)
plt.plot(x,bbvi_basic,label='bbvi_basic',alpha=0.8)
plt.plot(x,abbvi_cv,label='abbvi_cv',alpha=0.8)


plt.legend()
plt.grid()
plt.show()


a=1