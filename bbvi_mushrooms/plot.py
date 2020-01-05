import numpy as np
import matplotlib.pyplot as plt

bbvi_basic=[]
bbvi_cv=[]
abbvi_basic=[]

num=10
interval=10
iter_perEpoch=24


for i in range(num):
    bbvi_basic.append(np.load('./elbos/bbvi_basic/{:}.npy'.format(i)))
    bbvi_cv.append(np.load('./elbos/bbvi_cv/{:}.npy'.format(i)))
    abbvi_basic.append(np.load('./elbos/abbvi_basic/{:}.npy'.format(i)))
bbvi_basic=np.array(bbvi_basic)[:,0:120:4]
bbvi_cv=np.array(bbvi_cv)[:,0:120:4]
abbvi_basic=np.array(abbvi_basic)[:,0:120:4]

bbvi_basic_mean=np.mean(bbvi_basic,0)
bbvi_cv_mean=np.mean(bbvi_cv,0)
abbvi_basic_mean=np.mean(abbvi_basic,0)

bbvi_basic_std=np.std(bbvi_basic,axis=0)
bbvi_cv_std=np.std(bbvi_cv,axis=0)
abbvi_basic_std=np.std(abbvi_basic,axis=0)

x=np.linspace(1,len(bbvi_basic_mean),len(bbvi_basic_mean))*interval
plt.errorbar(x=x,y=bbvi_basic_mean,yerr=bbvi_basic_std,color='g',capsize=2,label='bbvi_basic')
plt.errorbar(x=x,y=bbvi_cv_mean,yerr=bbvi_cv_std,color='b',capsize=2,label='bbvi_cv')
plt.errorbar(x=x,y=abbvi_basic_mean,yerr=abbvi_basic_std,color='r',capsize=2,label='abbvi_basic')
plt.legend()
'''





for  i in range(num):
    plt.plot(bbvi_basic[i],color='g')
    plt.plot(bbvi_cv[i],color='b')
    plt.plot(abbvi_basic[i],color='r')
'''
plt.show()


