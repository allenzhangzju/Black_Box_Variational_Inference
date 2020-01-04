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
def elbo_evaluate(images,labels,para,dim,scale,revise,num_St):
    '''
    测试ELBO
    '''
    z_samples=sampleZ(para,dim,num_St)
    log_qs=ng_log_Qs(para,z_samples,dim)
    log_priors=ng_log_Priors(z_samples,dim)
    log_likelihoods=ng_log_Likelihoods(images,labels,z_samples,dim)
    elbo=log_likelihoods*revise+log_priors/scale-log_qs/scale
    avg=torch.sum(elbo)/num_St
    return avg

def nabla_F_Calc(images,labels,para,dim,num_S,scale,revise):
    '''
    计算梯度与其二范数，为了缩减abbvi的main代码长度
    主体结构和bbvi_basic的main相同
    '''
    gradients=torch.zeros((num_S,dim*2))
    z_samples=sampleZ(para,dim,num_S)
    log_qs=ng_log_Qs(para,z_samples,dim)
    log_priors=ng_log_Priors(z_samples,dim)
    log_likelihoods=ng_log_Likelihoods(images,labels,z_samples,dim)
    for s in range(len(z_samples)):
        gradients[s]=grad_log_Q(para,z_samples[s],dim)[0]
    elbo_temp=log_likelihoods*revise+log_priors/scale-log_qs/scale
    grad_temp=torch.matmul(torch.diag(elbo_temp),gradients)
    grad_d=torch.mean(grad_temp,0)
    G_pow2=torch.pow(grad_d.norm(),2)
    return grad_d,G_pow2



def nabla_F_cv_Calc(images,labels,para,dim,num_S,scale,revise):
    '''
    nabla_F_Calc的Control Variates版本
    '''
    gradients=torch.zeros((num_S,dim*2))
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
    grad_d=grads.mean(0)
    G_pow2=torch.pow(grad_d.norm(),2)
    return grad_d,G_pow2

@torch.no_grad()
def cvA_Calc(f,h,dim):
    '''
    Control Variates 的系数计算
    '''
    num_S=len(f)
    f_avg=f.mean(0)
    h_avg=h.mean(0)
    f_avgs=torch.Tensor(num_S,dim).copy_(f_avg.view(-1,dim))
    h_avgs=torch.Tensor(num_S,dim).copy_(h_avg.view(-1,dim))
    f0=f-f_avgs
    h0=h-h_avgs
    a=torch.diag(torch.matmul(f0.transpose(0,1),h0))/\
        torch.diag(torch.matmul(h0.transpose(0,1),h0))
    return torch.Tensor(num_S,dim).copy_(a.view(-1,dim))


def Delta_Calc(images,labels,para1,para0,eta,dim,num_S,M,scale):
    Delta=torch.zeros((M,dim*2))
    delta=(para1-para0).clone().detach().requires_grad_(False)
    A=torch.rand(M)
    for i in range(M):
        para_a=((1-A[i])*para0+A[i]*para1).clone().detach()
        Delta[i]=hessian_F_Calc(images,labels,para_a,delta,eta,dim,num_S,scale)
    avg=torch.mean(Delta,0)
    return avg

def hessian_F_Calc_approx(images,labels,para_a,delta,eta,dim,num_S,scale,revise):
    '''
    式（9）
    '''
    hessian_F=torch.zeros((num_S,dim*2))
    para=para_a.clone().detach().requires_grad_(True)
    gradients=torch.zeros((num_S,dim*2))
    z_samples=sampleZ(para,dim,num_S)
    log_qs=ng_log_Qs(para,z_samples,dim)
    log_priors=ng_log_Priors(z_samples,dim)
    log_likelihoods=ng_log_Likelihoods(images,labels,z_samples,dim)
    for s in range(num_S):
        gradients[s]=grad_log_Q(para,z_samples[s],dim)[0]
    elbo_temp=log_likelihoods*revise+log_priors/scale-log_qs/scale
    phi_eta=phi_eta_Calc_approx(para,z_samples,dim,delta,eta)
    for i in range(num_S):
        hessian_F[i]=elbo_temp[i]*phi_eta[i]+(elbo_temp[i]-1)*gradients[i]*torch.matmul(gradients[i],delta)
    avg=torch.mean(hessian_F,0)
    return avg
    
def hessian_F_Calc(images,labels,para_a,delta,dim,num_S,scale,revise):
    '''
    式（3） =  nabla^2 F * delta
    '''
    hessian_F=torch.zeros((num_S,dim*2))
    para=para_a.clone().detach().requires_grad_(True)
    gradients=torch.zeros((num_S,dim*2))
    z_samples=sampleZ(para,dim,num_S)
    log_qs=ng_log_Qs(para,z_samples,dim)
    log_priors=ng_log_Priors(z_samples,dim)
    log_likelihoods=ng_log_Likelihoods(images,labels,z_samples,dim)
    for s in range(num_S):
        gradients[s]=grad_log_Q(para,z_samples[s],dim)[0]
    elbo_temp=log_likelihoods*revise+log_priors/scale-log_qs/scale
    phi=phi_Calc(para,z_samples,dim,delta)
    for i in range(num_S):
        partA=(elbo_temp[i]-1)*gradients[i]*torch.matmul(gradients[i],delta)
        partB=elbo_temp[i]*phi[i]
        hessian_F[i]=partA+partB
    avg=torch.mean(hessian_F,0)
    return avg

def hessian_F_cv_Calc(images,labels,para_a,delta,dim,num_S,scale,revise):
    '''
    式（3）的Control Variates 版本
    '''
    f=torch.zeros((num_S,dim*2))
    h=torch.zeros((num_S,dim*2))
    para=para_a.clone().detach().requires_grad_(True)
    gradients=torch.zeros((num_S,dim*2))
    z_samples=sampleZ(para,dim,num_S)
    log_qs=ng_log_Qs(para,z_samples,dim)
    log_priors=ng_log_Priors(z_samples,dim)
    log_likelihoods=ng_log_Likelihoods(images,labels,z_samples,dim)
    for s in range(num_S):
        gradients[s]=grad_log_Q(para,z_samples[s],dim)[0]
    elbo_temp=log_likelihoods*revise+log_priors/scale-log_qs/scale
    phi=phi_Calc(para,z_samples,dim,delta)
    for i in range(num_S):
        h[i]=gradients[i]*torch.matmul(gradients[i],delta)+phi[i]
        f[i]=h[i]*elbo_temp[i]
    a=cvA_Calc(f,h,dim*2)
    results=f-torch.mul(a,h)
    avg=results.mean(0)
    return avg

def phi_eta_Calc_approx(para,z_samples,dim,delta,eta):
    '''
    近似计算 phi_eta(x,delta)，式（7）
    '''
    phi_eta=torch.zeros((len(z_samples),dim*2))
    para1=(para+eta*delta).clone().detach().requires_grad_(True)
    para0=(para-eta*delta).clone().detach().requires_grad_(True)
    for i in range(len(z_samples)):
        grad_para1=grad_log_Q(para1,z_samples[i],dim)[0]
        grad_para0=grad_log_Q(para0,z_samples[i],dim)[0]
        phi_eta[i]=(grad_para1-grad_para0)/(2*eta)
    return phi_eta

def phi_Calc(para,z_samples,dim,delta):
    '''
    精确计算 phi(x)=hessian logp(z|x) * delta
    '''
    para_leaf=para.clone().detach().requires_grad_(True)
    phis=torch.zeros((len(z_samples),dim*2))
    for  i in range(len(z_samples)):
        grad=torch.autograd.grad(_log_Q(para_leaf,z_samples[i],dim),\
            para_leaf,create_graph=True)
        a=torch.matmul(grad[0],delta)
        phi=torch.autograd.grad(a,para_leaf)
        phis[i]=phi[0]
    return phis
'''
----------------------------------------------------------------------
'''
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
    

@torch.no_grad()
def data_preprocess(data):
    '''
    数据预处理，包括/255和加偏置两个步骤
    '''
    images=data[0].view(-1,112)
    lens=len(images)
    labels=data[1].view(lens)
    images=torch.div(images.float(),1.0)
    images=torch.cat([images.float(),torch.ones((lens,1))],1)#补bias
    return images,labels

