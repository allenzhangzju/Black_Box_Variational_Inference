import numpy as np
import torch

def sampleZ(mu_s,log_sigma2_s,dim,size):
    std=torch.sqrt(torch.exp(log_sigma2_s))
    Z= torch.zeros((dim,size))
    for  i in range(size):
        temp=torch.normal(mu_s,std)
        Z[:,i]=temp
    return Z

def log_Q(mu_s,log_sigma2_s,z_samples):
    std=torch.sqrt(torch.exp(log_sigma2_s))
    normal=torch.distributions.normal.Normal(mu_s,std)
    return torch.sum(normal.log_prob(samples),0)

def log_P(images,labels,z_samples,dim):
    normal=torch.distributions.normal.Normal(torch.zeros(dim),torch.ones(dim))
    log_prior=torch.sum(normal.log_prob(z_samples),0)
    a=torch.matmul(images,z_samples)
    a=torch.matmul(torch.diag(labels),a)
    log_likelihood=torch.sum(torch.sigmoid(a),0)
    return log_likelihood+log_prior

