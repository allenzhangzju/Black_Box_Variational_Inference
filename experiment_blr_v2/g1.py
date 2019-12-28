import torch
from  torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from class_data_load import DatasetFromCSV

@torch.no_grad()
def ng_log_Likelihoods(images,labels,z_samples,dim):
    '''
    计算似然
    '''
    batch_size=len(labels)
    num_S=len(z_samples)
    a=torch.matmul(images,z_samples.transpose(0,1))
    b=torch.Tensor(batch_size,num_S).copy_(labels.view(batch_size,-1))
    c=torch.mul(a,b)
    log_likelihoods=torch.log(torch.sigmoid(c))
    Sum=torch.sum(log_likelihoods,0)
    return Sum

#z_hat 的维数
dim=28*28+1#这里+1是偏置
#默认设定z_hat
z_hat=torch.ones(dim)*0.1
#log_likelihood_avg
log_likelihood_avg=-5.7423849106


#数据集读取
transform=transforms.ToTensor()
train_data=DatasetFromCSV('./dataset/train_images_csv.csv','./dataset/train_labels_csv.csv',transforms=transform)
if __name__ == "__main__":
    images=torch.Tensor(train_data.images.values)
    lens=len(images)
    images=torch.div(images.float(),255)
    images=torch.cat([images,torch.ones((lens,1))],1)#补bias
    labels=(torch.Tensor(train_data.labels.values)).view(lens)
    log_likelihood_avg=ng_log_Likelihoods(images,labels,z_hat.view(1,dim),dim)/lens
    print(log_likelihood_avg.item())

