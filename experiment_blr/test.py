import numpy as np
import torch
from  torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from data_load import DatasetFromCSV
from functions import*

num_epochs=10
batchSize=3
S=5
dim=28*28+1

#读取数据
transform=transforms.ToTensor()
train_data=DatasetFromCSV('./train_images_csv.csv','./train_labels_csv.csv',transforms=transform)
train_loader=DataLoader(train_data,batch_size=batchSize,shuffle=False)
#定义变量，前28*28是权重w，最后一个是偏差项bias
mu_s=torch.zeros(dim,requires_grad=True)
log_sigma2_s=torch.zeros(dim,requires_grad=True)




for epoch in range(num_epochs):
    for i ,data in enumerate(train_loader):
        images=data[0].view(-1,28*28)
        labels=data[1]
        images=torch.div(images.float(),255)
        images=torch.cat([images,torch.ones((batchSize,1))],1)
        Z_s=sampleZ(mu_s,log_sigma2_s,dim,S).detach().numpy()
        a=1

