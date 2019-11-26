import numpy as np
import random
from scipy.stats import norm
from scipy.special import expit as sigmoid

class BBVI(object):
    def __init__(self,trainSource,testSource,featureDim,
    maxIteration,batchSize,sampleSize,stepScale,startPara,
    dataAccessing='RA',interval=100,testSampleNum=200):
        self._maxiteration=maxIteration#最大迭代次数
        self._batchSize=batchSize#batch大小
        self._dataAccessing=dataAccessing
        self._sampleSize=sampleSize#z采样数量
        self._stepScale=stepScale#步长
        self._interval=interval#测试间隔
        self.__batchCount=0#batch在CA获取方式下的标记点
        self._trainLabels,self._trainFeatures=self.__loading(trainSource,featureDim)
        self._testLabels,self._testFeatures=self.__loading(testSource,featureDim)
        self._dim=len(self._trainFeatures[0])#数据维数
        self._trainNum=len(self._trainLabels)#训练数据量
        self._testNum=len(self._testLabels)#测试数据量
        self._testSampleNum=testSampleNum
        '''
        '''
        self.epochs=np.array([batchSize*i/self._trainNum for i in range(maxIteration)])
        self.paras=[]
        self.paras.append(np.array(startPara))
        self.choosenTimes=np.zeros(self._trainNum)#记录训练样本被选中的次数
        self.accuracy=[]#测试项目，准确度
        self.avgloglikelihood=[]#测试项目，loss

    def __loading(self,source,dim):
        labels=[]
        features=[]
        dataNum=0
        '''
        读取数据
        '''
        with open(source,'r') as f:
            datas=f.readlines()
            dataNum=len(datas)
        '''
        转换格式
        '''
        for i in range(dataNum):
            line=datas[i]
            position=line.find(' ')
            label=eval(line[:position])
            feature=eval('{'+line[position+1:len(line)-2].replace(' ',',')+'}')
            temp=np.zeros(dim+1)
            temp[0]=1
            for key in feature.keys():
                temp[key]=1
            labels.append(label)
            features.append(temp)
        labels=np.array(labels)
        features=np.array(features)
        return labels,features

    def _getBatch(self):
        '''
        获取batch索引
        '''
        if(self._dataAccessing=='RA'):
            batches=random.sample(range(0,self._trainNum),self._batchSize)
        elif (self._dataAccessing=='CA'):
            start=self.__batchCount
            end=start+self._batchSize
            if(end<=len(self._trainLabels)):
                batches=list(range(start,end))
                self.__batchCount=end
            else:
                batches=list(range(start,len(self._trainLabels)))
                self.__batchCount=end-len(self._trainLabels)
                batches+=list(range(self.__batchCount))
        else:
            assert 0
        for index in batches:
            self.choosenTimes[index]+=1
        return batches

    def _logPCalc(self,z_sample):
        '''
        计算联合概率log_P
        '''
        minibatchs=self._getBatch()
        Z=np.array([self._trainLabels[index]*np.dot(self._trainFeatures[index],z_sample)\
            for index in minibatchs])
        log_likelihood=np.sum(np.log(sigmoid(Z)))*self._trainNum/self._batchSize
        log_prior=np.sum(norm.logpdf(z_sample,np.zeros(self._dim),np.ones(self._dim)))
        return log_likelihood+log_prior

    def _logQCalc(self,z_sample,mu,sigma2):
        '''
        计算变分分布log_q
        '''
        log_q=np.sum(norm.logpdf(z_sample,mu,np.sqrt(sigma2)))
        return log_q

    def _gradlogQCalc(self,z_sample,mu,sigma2):
        '''
        计算变分分布log_q的梯度
        '''
        score_mu=(z_sample-mu)/sigma2
        score_logsigma2=-0.5*np.ones(self._dim)+np.power((z_sample-mu),2)/(2*sigma2)
        return np.concatenate((score_mu,score_logsigma2))

    def _gradientCalc(self,z_sample,mu,sigma2):
        '''
        计算▽log_q*(log_p-log_q)
        '''
        gradientlogQ=self._gradlogQCalc(z_sample,mu,sigma2)
        log_p=self._logPCalc(z_sample)
        log_q=self._logQCalc(z_sample,mu,sigma2)
        return gradientlogQ*(log_p-log_q)

    def _paraTrans(self,i):
        '''
        将self.paras转换成正太分布的均值和方差
        '''
        mu=self.paras[i][0:self._dim]
        log_sigma2=self.paras[i][self._dim:]
        return mu,np.exp(log_sigma2)

    def _BBVI_basic(self):
        G=np.zeros((2*self._dim,2*self._dim))
        for i in range(self._maxiteration):
            '''
            获取需要的变分分布的均值和方差
            '''
            mu,sigma2=self._paraTrans(i)
            '''
            sample z
            '''
            z_samples=np.array([np.random.normal(mu,np.sqrt(sigma2))\
                    for i in range(self._sampleSize)])#梯度采样
            '''
            计算▽ELBO
            '''
            gradient=np.mean(np.array([self._gradientCalc(z_sample,mu,sigma2)\
                    for z_sample in z_samples]),axis=0)
            G+=np.outer(gradient,gradient)
            para_new=self.paras[i]+gradient*self._stepScale/np.sqrt(np.diag(G))
            '''
            append新值
            '''
            self.paras.append(np.array(para_new))
            '''
            test and print
            '''
            mu,sigma2=self._paraTrans(i+1)
            self._testCalc(i,mu,sigma2)
            self._informationPrint(i)
        self._postProcessing()

    def _informationPrint(self,i):
        if((self._testNum!=0 and  i%self._interval==0 and self._interval!=0)or i==self._maxiteration-1):
            print('第%d次迭代: '%(i))
            print('epochs: ',self.epochs[i])
            print('test_accuracy: ',self.accuracy[len(self.accuracy)-1])
            print('avgloglikelihood: ',self.avgloglikelihood[len(self.avgloglikelihood)-1])
            print('----------------------------------------------')

    def _postProcessing(self):
        '''
        后处理，list转array
        '''
        self.paras=np.array(self.paras)
        self.accuracy=np.array(self.accuracy)
        self.avgloglikelihood=np.array(self.avgloglikelihood)
        self.epochs=np.array(self.epochs)

    def __testCalc1(self,i,para):
        Z=[self._testLabels[j]*np.dot(self._testFeatures[j],para) for j in range(self._testNum)]
        Y=sigmoid(Z)
        return Y

    def _testCalc(self,i,mu,sigma2):
        if((self._testNum!=0 and  i%self._interval==0 and self._interval!=0)or i==self._maxiteration-1):
            z_samples=np.array([np.random.normal(mu,np.sqrt(sigma2))\
                    for i in range(self._testSampleNum)])
            results=np.array([self.__testCalc1(i,para) for para in z_samples])
            Y=np.mean(results,axis=0)
            T=Y*2
            accuracy=sum(T.astype(int))/self._testNum
            avgloglikelihood=-sum(np.log(Y))/self._testNum
            self.accuracy.append(accuracy)
            self.avgloglikelihood.append(avgloglikelihood)
