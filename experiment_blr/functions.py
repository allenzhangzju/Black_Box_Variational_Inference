import numpy as np
import torch

def sampleZ(mu_s,log_sigma2_s,dim):
    std=torch.sqrt(torch.exp(log_sigma2_s))
    return torch.normal(mu_s,std)

def log_Q(mu_s,log_sigma2_s,z_sample):
    std=torch.sqrt(torch.exp(log_sigma2_s))
    normal=torch.distributions.normal.Normal(mu_s,std)
    return torch.sum(normal.log_prob(z_sample),0)

def log_P(images,labels,z_sample,dim,num):
    normal=torch.distributions.normal.Normal(torch.zeros(dim),torch.ones(dim))
    log_prior=torch.sum(normal.log_prob(z_sample),0)
    a=torch.matmul(images,z_sample)
    log_likelihood=torch.sum(torch.log(torch.sigmoid(torch.mul(a,labels))),0)
    return log_likelihood+log_prior

def grad_log_Q(mu_s,log_sigma2_s,z_sample):
    pass

def accuracyCalc(mu_s,log_sigma2_s,test_data,dim):
    images=torch.tensor(test_data.images.values/255).float()
    images=torch.cat([images,torch.ones((len(images),1))],1)
    labels=torch.tensor(test_data.labels.values).view(len(images))
    a=torch.matmul(images,mu_s)
    accuracy=np.sum(np.round(torch.sigmoid(torch.mul(a,labels)).detach().numpy()))/len(labels)
    return accuracy
    
def mu1_varianceCalc(mu1):
    return np.var(np.array(mu1))