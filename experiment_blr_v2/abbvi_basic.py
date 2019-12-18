import numpy as np
import torch
from  torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from class_data_load import DatasetFromCSV
from functions import*
import os
'''
abbvi without any extension
'''
num_epochs=15
batchSize=256
num_S=20#训练的采样数量
dim=28*28+1#这里+1是偏置
eta=0.05#eta、k、w、c这四个参数是和论文对应的
k=0.2
w=1
c=1e7
M=10
num_St=1000#测试的采样数量
#读取数据
transform=transforms.ToTensor()
train_data=DatasetFromCSV('./dataset/train_images_csv.csv','./dataset/train_labels_csv.csv',transforms=transform)
test_data=DatasetFromCSV('./dataset/test_images_csv.csv','./dataset/test_labels_csv.csv',transforms=transform)
train_loader=DataLoader(train_data,batch_size=batchSize,shuffle=True)

#定义分布参数
para=torch.zeros(dim*2,requires_grad=True)
para[dim:]=torch.ones(dim)*(-1)


#需要储存结果
elbo_list=[]

#变量
G_pow2=None
grad_d=None
para_last=None

test_D=None
test_G=None

#开始迭代
for epoch in range(num_epochs):
    for i ,data in enumerate(train_loader):
        images,labels=data_preprocess(data)
        scale=len(train_loader)
        #ELBO evaluate
        elbo_list.append(elbo_evaluate(images,labels,para,dim,scale,num_St).item())
        #算法起始位置
        if(epoch==0 and i==0):
            grad_d,G_pow2=nabla_F_Calc(images,labels,para,dim,num_S,scale)
            continue
        #计算步长
        rho=k/(w+G_pow2)**(1/3)
        #迭代更新
        para_last=para.clone().detach()
        para.data+=rho*grad_d
        #计算bt
        b=c*rho*rho
        if b>1: b=1
        #计算nabla_F及二范数
        nabla_F,temp=nabla_F_Calc(images,labels,para,dim,num_S,scale)
        G_pow2+=temp
        #计算Delta
        Delta=Delta_Calc(images,labels,para,para_last,eta,dim,num_S,M,scale)
        #test_D=Delta.clone().detach().numpy()
        #test_G=grad_d.clone().detach().numpy()
        grad_d=(1-b)*(grad_d+Delta)+b*nabla_F
        print(b)
        #print information
        if 1:
            print('Epoch[{}/{}], step[{}/{}]'.format(\
                epoch+1,
                num_epochs,
                i+1,len(train_loader)))
            print('ELBO: {:.3f}\n'.format(\
                elbo_list[len(elbo_list)-1]))


if not os.path.exists('./result'):
    os.makedirs('./result')
result=np.array(elbo_list)
np.save('./result/abbvi_basic.npy',result)
