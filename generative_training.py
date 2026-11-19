"""Small, real trained generative mechanisms, not pretrained BERT/GPT/Stable Diffusion."""
import torch
from torch import nn
import numpy as np

torch.set_num_threads(1)


def train_denoising(steps=400, seed=18):
    torch.manual_seed(seed)
    # A two-cluster 2D distribution, not images.
    x=torch.randn(512,2)*.25+torch.where(torch.arange(512)[:,None]%2==0,torch.tensor([-1.,-1.]),torch.tensor([1.,1.]))
    T=24; beta=torch.linspace(.005,.09,T);alpha=1-beta;abar=alpha.cumprod(0)
    net=nn.Sequential(nn.Linear(3,48),nn.SiLU(),nn.Linear(48,48),nn.SiLU(),nn.Linear(48,2))
    opt=torch.optim.Adam(net.parameters(),lr=.003); losses=[]
    for _ in range(steps):
        ids=torch.randint(0,len(x),(96,));t=torch.randint(0,T,(96,));noise=torch.randn(96,2)
        a=abar[t,None];xt=a.sqrt()*x[ids]+(1-a).sqrt()*noise
        prediction=net(torch.cat([xt,t[:,None]/T],1));loss=((prediction-noise)**2).mean()
        opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
    # Actual reverse DDPM update. beta_tilde is used except at t=0.
    sample=torch.randn(180,2)
    net.eval()
    with torch.no_grad():
        for t in reversed(range(T)):
            eps=net(torch.cat([sample,torch.full((len(sample),1),t/T)],1))
            mean=(sample-beta[t]/torch.sqrt(1-abar[t])*eps)/torch.sqrt(alpha[t])
            variance=beta[t]*(1-abar[t-1])/(1-abar[t]) if t>0 else torch.tensor(0.)
            sample=mean+variance.sqrt()*torch.randn_like(sample) if t>0 else mean
    return x.numpy(),sample.numpy(),losses


def train_vae(steps=300, latent=1, seed=8):
    torch.manual_seed(seed)
    t=torch.linspace(0,6.283,240)
    x=torch.stack([torch.cos(t),torch.sin(t)],1)+.05*torch.randn(240,2)
    encoder=nn.Sequential(nn.Linear(2,24),nn.Tanh(),nn.Linear(24,latent*2))
    decoder=nn.Sequential(nn.Linear(latent,24),nn.Tanh(),nn.Linear(24,2))
    opt=torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()),lr=.01); losses=[]
    for _ in range(steps):
        mu,logvar=encoder(x).chunk(2,dim=1)
        z=mu+torch.randn_like(mu)*torch.exp(.5*logvar)
        reconstruction=((decoder(z)-x)**2).mean();kl=-.5*(1+logvar-mu.pow(2)-logvar.exp()).mean()
        loss=reconstruction+.03*kl;opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
    with torch.no_grad(): generated=decoder(torch.randn(180,latent)).numpy()
    return x.numpy(),generated,losses


def train_language(objective='causal', steps=80, seed=6):
    torch.manual_seed(seed)
    # Repeating modular arithmetic sequences over six symbols, with token 6 as MASK.
    starts=torch.arange(300)%6; tokens=(starts[:,None]+torch.arange(8)[None,:])%6
    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__(); self.embed=nn.Embedding(7,12); self.position=nn.Parameter(torch.randn(1,8,12)*.05)
            layer=nn.TransformerEncoderLayer(12,3,24,dropout=0,batch_first=True)
            self.encoder=nn.TransformerEncoder(layer,1,enable_nested_tensor=False); self.head=nn.Linear(12,6)
        def forward(self,x,causal=False):
            h=self.embed(x)+self.position[:,:x.shape[1],:]
            mask=torch.triu(torch.full((x.shape[1],x.shape[1]),float('-inf')),diagonal=1) if causal else None
            return self.head(self.encoder(h,mask=mask))
    net=TinyLM();opt=torch.optim.Adam(net.parameters(),lr=.012);losses=[]
    for _ in range(steps):
        train=tokens[:240]
        if objective=='causal':
            logits=net(train[:,:-1],True); target=train[:,1:];loss=nn.functional.cross_entropy(logits.reshape(-1,6),target.reshape(-1))
        else:
            masked=train.clone(); positions=torch.randint(0,8,(len(train),));target=train[torch.arange(len(train)),positions];masked[torch.arange(len(train)),positions]=6
            logits=net(masked)[torch.arange(len(train)),positions];loss=nn.functional.cross_entropy(logits,target)
        opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
    net.eval()
    with torch.no_grad():
        test=tokens[240:]
        if objective=='causal':
            pred=net(test[:,:-1],True).argmax(-1);acc=float((pred==test[:,1:]).float().mean())
        else:
            masked=test.clone();masked[:,3]=6;pred=net(masked)[:,3,:].argmax(-1);acc=float((pred==test[:,3]).float().mean())
    return {'objective':objective,'loss':losses,'held_out_token_accuracy':acc,'test_sequences':len(test),'note':'Synthetic periodic task, not a natural-language benchmark or pretrained model.'}
