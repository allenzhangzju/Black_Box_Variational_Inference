from sys import path
path.append('./')
import numpy as np
import torch
from  torch.utils.data.dataloader import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from class_data_load import DatasetFromCSV
from functions import*
import os
'''
find an appropriate num_St
'''
batchSize=120
dim=28*28+1#这里+1是偏置
num_St=2000#测试的采样数量
start_position=0#选取的测试数据的起始位置
#读取数据
transform=transforms.ToTensor()
train_data=DatasetFromCSV('./dataset/train_images_csv.csv','./dataset/train_labels_csv.csv',transforms=transform)
test_data=DatasetFromCSV('./dataset/test_images_csv.csv','./dataset/test_labels_csv.csv',transforms=transform)

images=torch.Tensor(train_data.images.values[start_position:start_position+batchSize])
labels=(torch.Tensor(train_data.labels.values[start_position:start_position+batchSize])).view(batchSize)
images=torch.div(images.float(),255)
images=torch.cat([images,torch.ones((batchSize,1))],1)#补bias
paras=np.load('./result_para/bbvi_cv.npy')
para=torch.Tensor(paras[1500])
scale=12223/batchSize
revise=1

elbos=[]
for i in range(20):
    elbo=elbo_evaluate(images,labels,para,dim,scale,revise,num_St)
    print(elbo)
    elbos.append(elbo.item())

plt.plot(elbos)
plt.show()
a=1
