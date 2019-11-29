import numpy as np
import torch

def sampleZ(mu_s,log_sigma2_s,dim):
    '''
    采样,返回一组向量
    '''
    std=torch.sqrt(torch.exp(log_sigma2_s))
    return torch.normal(mu_s,std)

def log_Q(mu_s,log_sigma2_s,z_sample):
    '''
    计算log q，返回一个标量
    '''
    std=torch.sqrt(torch.exp(log_sigma2_s))
    normal=torch.distributions.normal.Normal(mu_s,std)
    return torch.sum(normal.log_prob(z_sample),0)

def log_P(images,labels,z_sample,dim):
    '''
    计算log p（先验+似然），返回一个标量
    '''
    normal=torch.distributions.normal.Normal(torch.zeros(dim),torch.ones(dim))
    log_prior=torch.sum(normal.log_prob(z_sample),0)
    a=torch.matmul(images,z_sample)
    log_likelihood=torch.sum(torch.log(torch.sigmoid(torch.mul(a,labels))),0)
    return log_likelihood+log_prior

@torch.no_grad()
def rao_blackwellization_elbo(mu_s,log_sigma2_s,images,labels,z_sample,dim):
    '''
    计算 log p_i(x,z_s)-log q_i(z_s|lambda_i)，返回一个dim维度的向量
    参考Black Box Variational Inference 的公式（6）
    '''
    '''
    对于似然log_likelihood，其中的i不可拆，所以梯度的每个分量i这一部分是相同的
    对于先验log_prior，其中的i可拆，梯度的每个分量i这部分不同
    对于变分log_q，其中的i可拆，梯度的每个分量i这部分相同
    '''
    a=torch.matmul(images,z_sample)
    log_likelihood=torch.sum(torch.log(torch.sigmoid(torch.mul(a,labels))))*\
        torch.ones(len(z_sample))#这里的log_likelihood每个i都一样
    normal=torch.distributions.normal.Normal(torch.zeros(dim),torch.ones(dim))
    log_prior=normal.log_prob(z_sample)#这里的log_prior不同的i不一样
    log_joint=log_likelihood+log_prior#注：这里的log_joint依然是一个向量，不同的i不一样
    std=torch.sqrt(torch.exp(log_sigma2_s))
    normal=torch.distributions.normal.Normal(mu_s,std)
    log_q=normal.log_prob(z_sample)#这里的log_q是一个向量，不同的i不一样
    return log_joint-log_q

@torch.no_grad()
def control_variates_a(f,h,dim):
    '''
    计算control variates的系数，返回一个dim*2维度的向量
    参考Black Box Variational Inference 的公式（9)
    '''
    a=torch.zeros(dim)
    f1=f[0:dim]
    f2=f[dim:]
    h1=h[0:dim]
    h2=h[dim:]
    for i in range(dim):
        cov=torch.mean(f1[i]*h1[i])-torch.mean(f1[i])*torch.mean(h1[i])
        cov+=torch.mean(f2[i]*h2[i])-torch.mean(f2[i])*torch.mean(h2[i])
        a[i]=cov/(torch.var(h1[i])+torch.var(h2[i]))
    return torch.cat([a,a],0)


def accuracyCalc(mu_s,log_sigma2_s,test_data,dim):
    images=torch.tensor(test_data.images.values/255).float()
    images=torch.cat([images,torch.ones((len(images),1))],1)
    labels=torch.tensor(test_data.labels.values).view(len(images))
    a=torch.matmul(images,mu_s)
    accuracy=np.sum(np.round(torch.sigmoid(torch.mul(a,labels)).detach().numpy()))/len(labels)
    return accuracy
    
def mu1_varianceCalc(mu1):
    return np.var(np.array(mu1))

def data_preprocess(data):
    images=data[0].view(-1,28*28)
    lens=len(images)
    labels=data[1].view(lens)
    images=torch.div(images.float(),255)
    images=torch.cat([images,torch.ones((lens,1))],1)#补bias
    return images,labels
