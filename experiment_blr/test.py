import numpy as np
import torch
from  torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from data_load import DatasetFromCSV
from functions import*

N=12223
num_epochs=10
batchSize=128
S=5
dim=28*28+1
eta=0.2
#读取数据
transform=transforms.ToTensor()
train_data=DatasetFromCSV('./train_images_csv.csv','./train_labels_csv.csv',transforms=transform)
test_data=DatasetFromCSV('./test_images_csv.csv','./test_labels_csv.csv',transforms=transform)
train_loader=DataLoader(train_data,batch_size=batchSize,shuffle=False)

#定义变量，前28*28是权重w，最后一个是偏差项bias
mu_s=torch.zeros(dim,requires_grad=True)
log_sigma2_s=torch.zeros(dim,requires_grad=True)

#
elbo_list=[]
accuracy_list=[]

G=torch.zeros((dim*2,dim*2))
for epoch in range(num_epochs):
    for i ,data in enumerate(train_loader):
        images=data[0].view(-1,28*28)
        labels=data[1].view(len(images))
        images=torch.div(images.float(),255)
        images=torch.cat([images,torch.ones((len(labels),1))],1)#补bias
        gradient=torch.zeros(dim*2)
        elbo_avg=0
        mu1=[]
        for s in range(S):
            z_sample=sampleZ(mu_s,log_sigma2_s,dim)
            log_p=log_P(images,labels,z_sample,dim,N)
            log_q=log_Q(mu_s,log_sigma2_s,z_sample)
            log_q.backward()
            elbo=log_p-log_q
            gradient[0:dim]+=mu_s.grad*elbo
            gradient[dim:]+=log_sigma2_s.grad*elbo
            elbo_avg+=elbo
            mu1.append(((mu_s.grad[0])*elbo).item())
            mu_s.grad.data.zero_()
            log_sigma2_s.grad.data.zero_()
        elbo_avg/=S
        elbo_list.append(elbo_avg)
        gradient/=S
        G+=torch.matmul(gradient.view(dim*2,-1),gradient.view(-1,dim*2))
        rho=eta/torch.sqrt(torch.diag(G))
        mu_s.data+=torch.mul(rho[0:dim],gradient[0:dim])
        log_sigma2_s.data+=torch.mul(rho[dim:],gradient[dim:])
        accuracy=accuracyCalc(mu_s,log_sigma2_s,test_data,dim)
        accuracy_list.append(accuracy)
        variance=mu1_varianceCalc(mu1)
        print('{:.8f}, {:.8f}, {:.8f}'.format(elbo_avg,accuracy,variance))
        

