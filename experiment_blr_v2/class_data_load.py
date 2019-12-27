import numpy as np
import pandas as pd
from torch import optim,nn
import torch.nn.functional as func
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 

class DatasetFromCSV(Dataset):
    def __init__(self,image_path,label_path,transforms=None):
        self.images=pd.read_csv(image_path,header=None)
        self.labels=pd.read_csv(label_path,header=None)
        self.transforms=transforms
    
    def __getitem__(self,index):
        single_label=np.asarray(self.labels.iloc[index])
        single_image=np.asarray(self.images.iloc[index]).reshape(28,28)
        if self.transforms is not None:
            image_tensor=self.transforms(single_image)
            #label_tensor=self.transforms(single_label)
            return (image_tensor,single_label)
        return (single_image,single_label)
    
    def __len__(self):
        return len(self.images.index)
'''
transform=transforms.ToTensor()
train_data=DatasetFromCSV('./dataset/train_images_csv.csv','./dataset/train_labels_csv.csv',transforms=transform)
train_loader=DataLoader(train_data,batch_size=10,shuffle=True)
for i,data in enumerate(train_loader):
    images=data[0].view(-1,28,28)[0]
    print(data[1])
    plt.imshow(images)
    plt.show()

    a=1
b=1
'''