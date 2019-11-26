import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
import csv

train_set=torchvision.datasets.MNIST("./dataset",train=True,download=True,transform=transforms.ToTensor())
test_set=torchvision.datasets.MNIST("./dataset",train=False,download=True,transform=transforms.ToTensor())
train_images=train_set.data.view(-1,28*28).numpy()
train_labels=train_set.targets.numpy()

train_images_csv=[]
train_labels_csv=[]

for i in range(len(train_labels)):
    if(train_labels[i]==2):
        train_images_csv.append(train_images[i])
        train_labels_csv.append([-1])
    elif(train_labels[i]==7):
        train_images_csv.append(train_images[i])
        train_labels_csv.append([1])
    else:
        pass
train_images_csv=np.array(train_images_csv)
train_labels_csv=np.array(train_labels_csv)


f=open('./train_images_csv.csv','w',encoding='utf-8')
csv_writer=csv.writer(f)
csv_writer.writerows(train_images_csv)
f.close()
f=open('./train_labels_csv.csv','w',encoding='utf-8')
csv_writer=csv.writer(f)
csv_writer.writerows(train_labels_csv)
f.close()