import numpy as np
import torch


def ng_para_transfer(para,dim):
    '''
    参数转换
    '''
    log_sigma2=para[dim:].clone().detach().requires_grad_(False)
    mu=para[0:dim].clone().detach().requires_grad_(False)
    std=log_sigma2.exp().sqrt()
    return mu,std

@torch.no_grad()
def sampleZ(para,dim,num_S):
    '''
    采样
    '''
    mu,std=ng_para_transfer(para,dim)
    mu_s=torch.Tensor(num_S,dim).copy_(mu.view(-1,dim))
    std_s=torch.Tensor(num_S,dim).copy_(std.view(-1,dim))
    eps=torch.randn(num_S,dim)
    z_samples=mu_s+torch.mul(std_s,eps)
    return z_samples

@torch.no_grad()
def ng_log_Qs(para,z_samples,dim):
    '''
    计算log_q
    '''
    num_S=len(z_samples)
    mu,std=ng_para_transfer(para,dim)
    normal=torch.distributions.normal.Normal(mu,std)
    probs=torch.zeros(num_S)
    for i in range(num_S):
        probs[i]=torch.sum(normal.log_prob(z_samples[i]))
    return probs

@torch.no_grad()
def ng_log_Priors(z_samples,dim):
    '''
    计算先验
    '''
    num_S=len(z_samples)
    normal=torch.distributions.normal.Normal(torch.zeros(dim),torch.ones(dim))
    probs=torch.zeros(num_S)
    for i in range(num_S):
        probs[i]=torch.sum(normal.log_prob(z_samples[i]))
    return probs


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

'''
计算梯度
'''
def _log_Q(para,z_sample,dim):
    mu=para[0:dim]
    log_sigma2=para[dim:]
    std=torch.sqrt(torch.exp(log_sigma2))
    normal=torch.distributions.normal.Normal(mu,std)
    return torch.sum(normal.log_prob(z_sample))
def grad_log_Q(para,z_sample,dim):
    grad=torch.autograd.grad(_log_Q(para,z_sample,dim),para)
    return grad

@torch.no_grad()
def elbo_evaluate(images,labels,para,dim,scale,num_St):
    z_samples=sampleZ(para,dim,num_St)
    log_qs=ng_log_Qs(para,z_samples,dim)
    log_priors=ng_log_Priors(z_samples,dim)
    log_likelihoods=ng_log_Likelihoods(images,labels,z_samples,dim)
    elbo=log_likelihoods+log_priors/scale-log_qs/scale
    avg=torch.sum(elbo)/num_St
    return avg

@torch.no_grad()
def accuracyCalc(mu_s,log_sigma2_s,test_data,dim):
    '''
    在测试集上计算正确率
    注：这里为了简化计算，直接用了均值，没采样
    '''
    images=torch.tensor(test_data.images.values/255).float()
    images=torch.cat([images,torch.ones((len(images),1))],1)
    labels=torch.tensor(test_data.labels.values).view(len(images))
    a=torch.matmul(images,mu_s)
    accuracy=np.sum(np.round(torch.sigmoid(torch.mul(a,labels)).detach().numpy()))/len(labels)
    return accuracy
    
def mu1_varianceCalc(mu1):
    return np.var(np.array(mu1))

@torch.no_grad()
def data_preprocess(data):
    '''
    数据预处理，包括/255和加偏置两个步骤
    '''
    images=data[0].view(-1,28*28)
    lens=len(images)
    labels=data[1].view(lens)
    images=torch.div(images.float(),255)
    images=torch.cat([images,torch.ones((lens,1))],1)#补bias
    return images,labels

def Delta(images,labels,M,mu_1,log_sigma2_1,mu_0,log_sigma2_0,S,sizaA,dim):
    grad=torch.zeros(dim*2)
    A=torch.rand(sizaA)
    for a in A:
        with torch.no_grad():
            mu=(1-a)*mu_0+a*mu_1
            log_sigma2=(1-a)*log_sigma2_0+a*log_sigma2_1
        hessian=Hessian(images,labels,M,mu,log_sigma2,S,dim)*M*M
        grad+=torch.matmul(hessian,torch.cat([(mu_1-mu_0),(log_sigma2_1-log_sigma2_0)]))
    result=grad/sizaA
    return result

def Hessian_log_Q(M,mu,log_sigma2,z_sample,dim):
    hessian=torch.zeros((dim*2,dim*2))
    para_leaf=torch.tensor(torch.cat([mu,log_sigma2]),requires_grad=True)
    grad_para=torch.autograd.grad(log_Q(para_leaf[0:dim],para_leaf[dim:],z_sample,M),\
        para_leaf,create_graph=True)
    i=0
    for anygrad in grad_para[0]:
        temp=torch.autograd.grad(anygrad,para_leaf,retain_graph=True)[0]
        hessian[i,:]=temp.view(-1,dim*2)
        i+=1
    return hessian




def Hessian(images,labels,M,mu,log_sigma2,S,dim):
    mu_t=torch.tensor(mu,requires_grad=True)
    log_sigma2_t=torch.tensor(log_sigma2,requires_grad=True)
    result=torch.zeros((dim*2,dim*2))
    for i in range(S):
        z_sample=sampleZ(mu_t,log_sigma2_t,dim)
        log_q=log_Q(mu_t,log_sigma2_t,z_sample,M)
        with torch.no_grad():
            f=log_P(images,labels,z_sample,dim,M)+log_q
        log_q.backward()
        grad_para=torch.cat([mu_t.grad,log_sigma2_t.grad])
        mu_t.grad.zero_()
        log_sigma2_t.grad.zero_()
        result+=f*Hessian_log_Q(M,mu_t,log_sigma2_t,z_sample,dim)+\
            torch.matmul(grad_para.view(dim*2,-1),grad_para.view(-1,dim*2))*(f-1)
    result/=S

    return result
