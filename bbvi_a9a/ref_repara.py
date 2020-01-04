import numpy as np
import torch
from  torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from class_data_load import DatasetFromCSV
from functions import*
import os

def elbo_repara(images,labels,para,num_S,dim,scale,revise):
    mu=para[0:dim]
    std=torch.sqrt(torch.exp(para[dim:]))
    elbo=torch.zeros(num_S)
    for i in range(num_S):
        eps=torch.randn_like(std)
        z=mu+eps*std
        normal1=torch.distributions.normal.Normal(mu,std)
        log_q=torch.sum(normal1.log_prob(z))
        normal2=torch.distributions.normal.Normal(torch.zeros(dim),torch.ones(dim))
        log_prior=torch.sum(normal2.log_prob(z))
        a=torch.matmul(images,z)
        log_likelihood=torch.sum(torch.log(torch.sigmoid(torch.mul(a,labels))))
        elbo[i]=log_likelihood*revise+log_prior/scale-log_q/scale
    elbo_avg=elbo.mean()
    return elbo_avg
'''
ref repara
'''
num_epochs=50
batchSize=1000
num_S=5#训练的采样数量
dim=123+1
eta=0.1#步长
num_St=1000#测试的采样数量
#读取数据
transform=transforms.ToTensor()
train_data=DatasetFromCSV('./dataset/train_images_csv.csv','./dataset/train_labels_csv.csv',transforms=transform)
test_data=DatasetFromCSV('./dataset/test_images_csv.csv','./dataset/test_labels_csv.csv',transforms=transform)
train_loader=DataLoader(train_data,batch_size=batchSize,shuffle=True)

#定义分布参数
para=torch.zeros(dim*2,requires_grad=True)
#para[dim:]=torch.ones(dim)*(-1)
scale=32561/batchSize


#需要储存结果
elbo_list=[]
para_list=[]

#AdaGrad
G=torch.zeros((dim*2,dim*2))

#开始迭代
for epoch in range(num_epochs):
    for i ,data in enumerate(train_loader):
        images,labels=data_preprocess(data)
        revise=batchSize/len(images)
        #过程变量
        gradients=torch.zeros((num_S,dim*2))
        #ELBO evaluate & record para
        if (epoch*len(train_loader)+i)%10==0:
            para_list.append(para.clone().detach().numpy())
            elbo_list.append(elbo_evaluate(images,labels,para,dim,scale,revise,num_St).item())
        #算法起始位置
        grad_avg=torch.autograd.grad(elbo_repara(images,labels,para,num_S,dim,scale,revise),para)[0]
        G+=torch.matmul(grad_avg.view(dim*2,-1),grad_avg.view(-1,dim*2))
        rho=eta/torch.sqrt(torch.diag(G))
        para.data+=rho*grad_avg
        #print information
        if (epoch*len(train_loader)+i)%10==0:
            print('Epoch[{}/{}], step[{}/{}]'.format(\
                epoch+1,
                num_epochs,
                i+1,len(train_loader)))
            print('ELBO: {:.3f}\n'.format(\
                elbo_list[len(elbo_list)-1]))


if not os.path.exists('./result_elbo'):
    os.makedirs('./result_elbo')
result=np.array(elbo_list)
np.save('./result_elbo/ref_repara.npy',result)


if not os.path.exists('./result_para'):
    os.makedirs('./result_para')
result=np.array(para_list)
np.save('./result_para/ref_repara.npy',result)





