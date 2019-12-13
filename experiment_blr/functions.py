import numpy as np
import torch

@torch.no_grad()
def sampleZ(mu_s,log_sigma2_s,dim):
    '''
    采样,返回一组向量
    '''
    std=torch.sqrt(torch.exp(log_sigma2_s))
    return torch.normal(mu_s,std)

def log_Q(mu_s,log_sigma2_s,z_sample,M):
    '''
    计算log q，返回一个标量
    '''
    std=torch.sqrt(torch.exp(log_sigma2_s))
    normal=torch.distributions.normal.Normal(mu_s,std)
    return torch.sum(normal.log_prob(z_sample),0)/M

def log_P(images,labels,z_sample,dim,M):
    '''
    计算log p（先验+似然），返回一个标量
    '''
    normal=torch.distributions.normal.Normal(torch.zeros(dim),torch.ones(dim))
    log_prior=torch.sum(normal.log_prob(z_sample),0)
    a=torch.matmul(images,z_sample)
    log_likelihood=torch.sum(torch.log(torch.sigmoid(torch.mul(a,labels))),0)
    return log_likelihood+log_prior/M

@torch.no_grad()
def grad_log_Q(mu_s,log_sigma2_s,z_sample):
    sigma2=torch.exp(log_sigma2_s)
    grad_mu=(z_sample-mu_s)/sigma2
    grad_sigma=((-1/(2*sigma2))+(grad_mu**2/2))*sigma2
    return torch.cat([grad_mu,grad_sigma],0)

def elbo_repara(images,labels,mu_s,log_sigma2_s,dim,M):
    '''
    用于reparameterzie方法的elbo计算，返回一个标量
    '''
    std=torch.sqrt(torch.exp(log_sigma2_s))
    eps=torch.randn_like(std)
    z=mu_s+eps*std
    return log_P(images,labels,z,dim,M)-log_Q(mu_s,log_sigma2_s,z,M)

@torch.no_grad()
def rao_blackwellization_elbo(mu_s,log_sigma2_s,images,labels,z_sample,dim,M):
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
    normal1=torch.distributions.normal.Normal(torch.zeros(dim),torch.ones(dim))
    log_prior=normal1.log_prob(z_sample)/M#这里的log_prior不同的i不一样
    log_joint=log_likelihood+log_prior#注：这里的log_joint依然是一个向量，不同的i不一样
    std=torch.sqrt(torch.exp(log_sigma2_s))
    normal2=torch.distributions.normal.Normal(mu_s,std)
    log_q=normal2.log_prob(z_sample)#这里的log_q是一个向量，不同的i不一样
    return log_joint-log_q/M

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
