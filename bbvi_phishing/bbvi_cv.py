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
bbvi with Control Variates
'''
num_epochs=50
batchSize=450
num_S=5#训练的采样数量
dim=68+1
eta=0.3#步长
num_St=5000#测试的采样数量
#读取数据
transform=transforms.ToTensor()
train_data=DatasetFromCSV('./dataset/train_images_csv.csv','./dataset/train_labels_csv.csv',transforms=transform)
#test_data=DatasetFromCSV('./dataset/test_images_csv.csv','./dataset/test_labels_csv.csv',transforms=transform)
train_loader=DataLoader(train_data,batch_size=batchSize,shuffle=True)

#定义分布参数
para=torch.zeros(dim*2,requires_grad=True)
#para[dim:]=torch.ones(dim)*(-1)
scale=11055/batchSize


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
        z_samples=sampleZ(para,dim,num_S)
        log_qs=ng_log_Qs(para,z_samples,dim)
        log_priors=ng_log_Priors(z_samples,dim)
        log_likelihoods=ng_log_Likelihoods(images,labels,z_samples,dim)
        for s in range(len(z_samples)):
            gradients[s]=grad_log_Q(para,z_samples[s],dim)[0]
        elbo_temp=log_likelihoods*revise+log_priors/scale-log_qs/scale
        f=torch.matmul(torch.diag(elbo_temp),gradients)
        h=gradients
        #Control Variates
        a=cvA_Calc(f,h,dim*2)
        grads=f-torch.mul(a,h)
        grad_avg=grads.mean(0)
        G+=torch.matmul(grad_avg.view(dim*2,-1),grad_avg.view(-1,dim*2))
        rho=eta/torch.sqrt(torch.diag(G))
        update=rho*grad_avg
        para.data+=update
        print(torch.median(update.abs()),torch.max(update.abs()))
        #print information
        if (epoch*len(train_loader)+i)%10==0:
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
np.save('./result_elbo/bbvi_cv.npy',result)


if not os.path.exists('./result_para'):
    os.makedirs('./result_para')
result=np.array(para_list)
np.save('./result_para/bbvi_cv.npy',result)
'''
result=np.array(elbo_list)
np.save(path,result)