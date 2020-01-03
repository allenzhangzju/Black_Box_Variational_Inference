# -*- coding: utf-8 -*-
import numpy as np
import torch
from  torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from class_data_load import DatasetFromCSV
from functions import*
import os
import sys
path=sys.argv[1]
'''
abbvi without any extension
'''
num_epochs=50
batchSize=500
num_S=5#训练的采样数量
dim=28*28+1#这里+1是偏置
#eta=0.05#eta、k、w、c这四个参数是和论文对应的
#k=0.4  w=8e10  c=0.7e8
#k=0.3  w=1e10  c=0.7e8
k=0.3
w=1e10
c=0.7e8
M=10
num_St=1000#测试的采样数量
#读取数据
transform=transforms.ToTensor()
train_data=DatasetFromCSV('./dataset/train_images_csv.csv','./dataset/train_labels_csv.csv',transforms=transform)
test_data=DatasetFromCSV('./dataset/test_images_csv.csv','./dataset/test_labels_csv.csv',transforms=transform)
train_loader=DataLoader(train_data,batch_size=batchSize,shuffle=True)

#定义分布参数
para=torch.zeros(dim*2,requires_grad=True)
#para[dim:]=torch.ones(dim)*(-1)
scale=12000/batchSize


#需要储存结果
elbo_list=[]
para_list=[]

#变量
G_pow2=None
grad_d=None
para_last=None


#开始迭代
for epoch in range(num_epochs):
    for i ,data in enumerate(train_loader):
        images,labels=data_preprocess(data)
        revise=batchSize/len(images)
        #ELBO evaluate & record para
        #para_list.append(para.clone().detach().numpy())
        if (epoch*len(train_loader)+i)%10==0:
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
        Delta_temp=torch.zeros((M,dim*2))
        delta=(para-para_last).clone().detach().requires_grad_(False)
        A=torch.rand(M)
        for j in range(M):
            para_a=((1-A[j])*para_last+A[j]*para).clone().detach()
            Delta_temp[j]=hessian_F_Calc(images,labels,para_a,delta,dim,num_S,scale,revise)
        Delta=Delta_temp.mean(0)
        #************************************************************************************
        grad_d=(1-b)*(grad_d+Delta)+b*nabla_F
        print(b,torch.median(update.abs()),torch.max(update.abs()))
        #print information
        if  (epoch*len(train_loader)+i)%10==0:
            print('Epoch[{}/{}], step[{}/{}]'.format(\
                epoch+1,
                num_epochs,
                i+1,len(train_loader)))
            print('ELBO: {:.3f}\n'.format(\
                elbo_list[len(elbo_list)-1]))

'''
if not os.path.exists('./result_elbo'):
    os.makedirs('./result_elbo')
result=np.array(elbo_list)
np.save('./result_elbo/abbvi_basic.npy',result)


if not os.path.exists('./result_para'):
    os.makedirs('./result_para')
result=np.array(para_list)
np.save('./result_para/abbvi_basic.npy',result)
'''
result=np.array(elbo_list)
np.save(path,result)