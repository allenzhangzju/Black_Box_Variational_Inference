import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


bbvi_cv=np.load('./test/batchSize/elbos3000.npy')

plt.plot(bbvi_cv,label='bbvi_cv',alpha=0.8)



plt.legend()
plt.grid()
plt.show()


a=1