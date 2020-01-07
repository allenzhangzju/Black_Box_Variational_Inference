import numpy as np
import torch
from  torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from functions import*
import os
'''
bbvi with Control Variates
'''
num_epochs=1
batchSize=100
num_S=5#训练的采样数量
dim=1000000+1
eta=0.05#步长
num_St=100#测试的采样数量
#读取数据
train_index=np.linspace(0,999999,1000000)
with open('./dataset/criteo-train-sub1000000.txt','r') as f:
    train_datas=f.readlines()
train_loader=DataLoader(train_index,batch_size=batchSize,shuffle=True)

#定义分布参数
para=torch.zeros(dim*2,requires_grad=True)
para[dim:]=torch.ones(dim)*(-1)
scale=1000000/batchSize
G=torch.zeros(dim*2)


#需要储存结果
elbo_list=[]
para_list=[]


#开始迭代
for epoch in range(num_epochs):
    for i ,data_index in enumerate(train_loader):
        labels,images=data_preprocess(train_datas,data_index,dim)
        revise=batchSize/len(images)
        #过程变量
        gradients=torch.zeros((num_S,dim*2))
        #ELBO evaluate & record para
        para_list.append(para.clone().detach().numpy())
        elbo_list.append(elbo_evaluate(images,labels,para,dim,scale,revise,num_St).item())
        #算法起始位置
        z_samples=sampleZ(para,dim,num_S)
        log_qs=ng_log_Qs(para,z_samples,dim)
        log_priors=ng_log_Priors(z_samples,dim)
        log_likelihoods=ng_log_Likelihoods(images,labels,z_samples,dim)
        for s in range(len(z_samples)):
            gradients[s]=grad_log_Q(para,z_samples[s],dim)[0]
        elbo_temp=log_likelihoods*revise+log_priors/scale-log_qs/scale
        f=torch.matmul(torch.diag(elbo_temp),gradients)
        h=gradients
        #Control Variates
        a=cvA_Calc(f,h,dim*2)
        grads=f-torch.mul(a,h)
        grad_avg=grads.mean(0)
        G+=grad_avg*grad_avg
        rho=eta/torch.sqrt(G)
        para.data+=rho*grad_avg
        #print information
        if 1:
            print('Epoch[{}/{}], step[{}/{}]'.format(\
                epoch+1,
                num_epochs,
                i+1,len(train_loader)))
            print('ELBO: {:.3f}\n'.format(\
                elbo_list[len(elbo_list)-1]))


if not os.path.exists('./result_elbo'):
    os.makedirs('./result_elbo')
result=np.array(elbo_list)
np.save('./result_elbo/bbvi_cv.npy',result)


if not os.path.exists('./result_para'):
    os.makedirs('./result_para')
result=np.array(para_list)
np.save('./result_para/bbvi_cv.npy',result)
