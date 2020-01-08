import numpy as np
import torch
from  torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from functions import*
import os
'''
abbvi without any extension
'''
num_epochs=1
batchSize=500
num_S=5#训练的采样数量
dim=1000000+1
num_St=100#测试的采样数量
#eta=0.05#eta、k、w、c这四个参数是和论文对应的
k=1
w=5e13
c=1.3e9
M=10
num_St=100#测试的采样数量
interval=20
#读取数据
train_index=np.linspace(0,999999,1000000)
with open('./dataset/criteo-train-sub1000000.txt','r') as f:
    train_datas=f.readlines()
train_loader=DataLoader(train_index,batch_size=batchSize,shuffle=True)

#定义分布参数
para=torch.zeros(dim*2,requires_grad=True)
#para[dim:]=torch.ones(dim)*(-1)
scale=1000000/batchSize
G=torch.zeros(dim*2)


#需要储存结果
elbo_list=[]
para_list=[]

#变量
G_pow2=None
grad_d=None
para_last=None


#开始迭代
for epoch in range(num_epochs):
    for i ,data_index in enumerate(train_loader):
        labels,images=data_preprocess(train_datas,data_index,dim)
        revise=batchSize/len(images)
        #ELBO evaluate & record para
        if i==len(train_loader)-1:
            para_list.append(para.clone().detach().numpy())
        if (epoch*len(train_loader)+i)%interval==0:
            elbo_list.append(elbo_evaluate(images,labels,para,dim,scale,revise,num_St).item())
        #算法起始位置
        if(epoch==0 and i==0):
            grad_d,G_pow2=nabla_F_Calc(images,labels,para,dim,num_S,scale,revise)
            continue
        #计算步长
        rho=k/(w+G_pow2)**(1/3)
        #迭代更新
        para_last=para.clone().detach()
        update=rho*grad_d
        para.data+=update
        #计算bt
        b=c*rho*rho
        if b>1: b=1
        #计算nabla_F及二范数
        nabla_F,temp=nabla_F_Calc(images,labels,para,dim,num_S,scale,revise)
        G_pow2+=temp
        #计算Delta  **************************************************************************
        Delta_temp=torch.zeros(dim*2)
        delta=(para-para_last).clone().detach().requires_grad_(False)
        A=torch.rand(M)
        for j in range(M):
            para_a=((1-A[j])*para_last+A[j]*para).clone().detach()
            Delta_temp+=hessian_F_Calc(images,labels,para_a,delta,dim,num_S,scale,revise)
        Delta=Delta_temp/M
        #************************************************************************************
        grad_d=(1-b)*(grad_d+Delta)+b*nabla_F
        print(b,torch.median(update.abs()),torch.max(update.abs()))
        #print information
        if (epoch*len(train_loader)+i)%interval==0:
            print('Epoch[{}/{}], step[{}/{}]'.format(\
                epoch+1,
                num_epochs,
                i+1,len(train_loader)))
            print('ELBO: {:.3f}\n'.format(\
                elbo_list[len(elbo_list)-1]))


if not os.path.exists('./result_elbo'):
    os.makedirs('./result_elbo')
result=np.array(elbo_list)
np.save('./result_elbo/abbvi_basic.npy',result)


if not os.path.exists('./result_para'):
    os.makedirs('./result_para')
result=np.array(para_list)
np.save('./result_para/abbvi_basic.npy',result)
