import numpy as np
import torch
from  torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from data_load import DatasetFromCSV
from functions import*
import os
'''
bbvi with Control Variates only
'''
num_epochs=15
batchSize=256
S=20
dim=28*28+1
eta=0.1
#读取数据
transform=transforms.ToTensor()
train_data=DatasetFromCSV('./train_images_csv.csv','./train_labels_csv.csv',transforms=transform)
test_data=DatasetFromCSV('./test_images_csv.csv','./test_labels_csv.csv',transforms=transform)
train_loader=DataLoader(train_data,batch_size=batchSize,shuffle=True)

#定义变量，前28*28是权重w，最后一个是偏差项bias
mu_s=torch.zeros(dim,requires_grad=True)
log_sigma2_s=torch.zeros(dim,requires_grad=True)

#需要储存结果
elbo_list=[]
variance_list=[]
accuracy_list=[]

#AdaGrad
G=torch.zeros((dim*2,dim*2))

#开始迭代
for epoch in range(num_epochs):
    for i ,data in enumerate(train_loader):
        images,labels=data_preprocess(data)
        M=len(train_loader)
        #过程变量
        f=torch.zeros((dim*2,S))
        h=torch.zeros((dim*2,S))
        a_para=torch.ones(dim*2)
        gradient=torch.zeros((dim*2,S))
        elbo=torch.zeros(S)
        mu1=np.zeros(S)
        #采样
        for s in range(S):
            z_sample=sampleZ(mu_s,log_sigma2_s,dim)
            log_p=log_P(images,labels,z_sample,dim,M)
            log_q=log_Q(mu_s,log_sigma2_s,z_sample,M)
            log_q.backward()#这里用自动求导
            with torch.no_grad():
                elbo[s]=log_p-log_q#这里记录elbo
                h[0:dim,s]=mu_s.grad*M#这里记录h
                h[dim:,s]=log_sigma2_s.grad*M
                f[0:dim,s]=h[0:dim,s]*elbo[s]#这里记录梯度
                f[dim:,s]=h[dim:,s]*elbo[s]
            mu_s.grad.data.zero_()#清除梯度，为准备下一次迭代
            log_sigma2_s.grad.data.zero_()
        with torch.no_grad():
            a_para=control_variates_a(f,h,dim)
            gradient=f-torch.matmul(torch.diag(a_para),h)
            mu1=gradient[0].detach().numpy()#这里记录μ1的梯度
            grad=gradient.mean(1)#求梯度均值
            G+=torch.matmul(grad.view(dim*2,-1),grad.view(-1,dim*2))#AdaGrad
            rho=eta/torch.sqrt(torch.diag(G))#AdaGrad
        mu_s.data+=torch.mul(rho[0:dim],grad[0:dim])#step
        log_sigma2_s.data+=torch.mul(rho[dim:],grad[dim:])#step
        elbo_list.append(np.mean(elbo.detach().numpy()))#求elbo的均值加入list
        accuracy_list.append(accuracyCalc(mu_s,log_sigma2_s,test_data,dim))#在测试集上计算accuracy
        variance_list.append(mu1_varianceCalc(mu1))#计算μ1的方差
        if (i+1)%10==0 or i<=10 and epoch==0:
            print('Epoch[{}/{}], step[{}/{}]'.format(\
                epoch+1,
                num_epochs,
                i+1,len(train_loader)))
            print('ELBO: {:.3f}, acc: {:.3f}, Var: {:.3f}\n'.format(\
                elbo_list[len(elbo_list)-1],
                accuracy_list[len(accuracy_list)-1],
                variance_list[len(variance_list)-1]))
        
if not os.path.exists('./result'):
    os.makedirs('./result')
result=np.array([elbo_list,variance_list,accuracy_list])
np.save('./result/bbvi_cv.npy',result)
