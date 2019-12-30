import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

bbvi_basic=np.load('./result_elbo/bbvi_basic.npy')
bbvi_cv=np.load('./result_elbo/bbvi_cv.npy')
abbvi_basic=np.load('./result_elbo/abbvi_basic.npy')
repara=np.load('./result_elbo/ref_repara.npy')
abbvi_cv=np.load('./result_elbo/abbvi_cv.npy')
#test=np.load('./result_para/abbvi_basic.npy')



plt.plot(repara,label='repara',alpha=0.8)
plt.plot(abbvi_basic,label='abbvi_basic',alpha=0.8)
plt.plot(bbvi_basic,label='bbvi_basic',alpha=0.8)
plt.plot(bbvi_cv,label='bbvi_cv',alpha=0.8)
#plt.plot(abbvi_cv,label='abbvi_cv',alpha=0.8)




plt.legend()
plt.grid()
plt.show()


a=1