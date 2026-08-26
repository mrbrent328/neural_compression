#!/usr/bin/env python3
import json, math, random, time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SEED=42; N_SAMPLES=2500; ROWS=64; COLS=64; LATENT_DIM=2
TRAIN_FRACTION=.8; EPOCHS=2000; BATCH_SIZE=128; LEARNING_RATE=2e-3
WEIGHT_DECAY=1e-5; NOISE_STD=.03; OUTPUT_DIR=Path('synthetic_latent_v1')

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def make_patterns(rows, cols, device):
    y=torch.linspace(-1,1,rows,device=device).view(-1,1); x=torch.linspace(-1,1,cols,device=device).view(1,-1)
    A=torch.exp(-((x+.35)**2+(y-.25)**2)/.30)
    B=torch.sin(2.5*x)*torch.cos(1.7*y)
    C=torch.sin(7*x+1.2*y)*torch.exp(-(x*x+y*y)/.8)
    D=A*B
    patterns=torch.stack([A,B,C,D]); patterns=patterns/patterns.flatten(1).std(dim=1).view(-1,1,1)
    return patterns

def generate_dataset():
    patterns=make_patterns(ROWS,COLS,'cpu')
    factors=torch.empty(N_SAMPLES,4); factors.uniform_(-1,1)
    a=factors[:,0].view(-1,1,1); b=factors[:,1].view(-1,1,1); c=factors[:,2].view(-1,1,1); d=factors[:,3].view(-1,1,1)
    tensors=a*patterns[0]+b*patterns[1]+c*patterns[2]+d*a*b*patterns[3]
    tensors += torch.randn_like(tensors)*NOISE_STD
    tensors=(tensors-tensors.mean())/tensors.std()
    return tensors,factors,patterns

class TinyAutoencoder(nn.Module):
    def __init__(self,input_dim,latent_dim=2):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(input_dim,256),nn.LayerNorm(256),nn.GELU(),nn.Linear(256,64),nn.GELU(),nn.Linear(64,latent_dim))
        self.decoder=nn.Sequential(nn.Linear(latent_dim,64),nn.GELU(),nn.Linear(64,256),nn.GELU(),nn.Linear(256,input_dim))
    def encode(self,x): return self.encoder(x)
    def decode(self,z): return self.decoder(z)
    def forward(self,x): z=self.encode(x); return self.decode(z),z

def normalized_mse(pred,target):
    mse=torch.mean((pred-target)**2); var=torch.mean((target-target.mean())**2); return (mse/(var+1e-12)).item()

def pearson_corr(x,y):
    x=np.asarray(x)-np.mean(x); y=np.asarray(y)-np.mean(y); denom=np.sqrt(np.sum(x*x)*np.sum(y*y)); return 0.0 if denom==0 else float(np.sum(x*y)/denom)

def correlation_matrix(latents,factors):
    names=['factor_A','factor_B','factor_C','interaction_strength']; return {f'z{i+1}':{n:pearson_corr(latents[:,i],factors[:,j]) for j,n in enumerate(names)} for i in range(latents.shape[1])}

def save_training_curve(h,path):
    plt.figure(figsize=(9,5)); plt.plot(h['epoch'],h['train_mse'],label='train'); plt.plot(h['epoch'],h['test_mse'],label='test'); plt.xlabel('Epoch'); plt.ylabel('Normalized MSE'); plt.title('Synthetic latent experiment'); plt.legend(); plt.tight_layout(); plt.savefig(path,dpi=150); plt.close()

def save_latent_scatter(latents,factors,path):
    fig,axes=plt.subplots(2,2,figsize=(10,9)); names=['factor A','factor B','factor C','interaction strength']
    for ax,idx,name in zip(axes.flat,range(4),names):
        sc=ax.scatter(latents[:,0],latents[:,1],c=factors[:,idx],s=8,alpha=.6); ax.set_xlabel('z1'); ax.set_ylabel('z2'); ax.set_title(name); fig.colorbar(sc,ax=ax)
    fig.suptitle('Learned 2D latent colored by known generating factors'); fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig)

def main():
    seed_everything(SEED); OUTPUT_DIR.mkdir(parents=True,exist_ok=True); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('='*80); print('KNOWN-FACTOR SYNTHETIC LATENT EXPERIMENT v1'); print('='*80)
    print(f'Device: {device}\nSamples: {N_SAMPLES}\nTensor: {ROWS} x {COLS}\nLatent: {LATENT_DIM}\nEpochs: {EPOCHS}\nNoise std: {NOISE_STD}\n')
    tensors,factors,patterns=generate_dataset(); flat=tensors.flatten(1); split=int(N_SAMPLES*TRAIN_FRACTION)
    train_x,test_x=flat[:split],flat[split:]; train_factors,test_factors=factors[:split],factors[split:]
    loader=DataLoader(TensorDataset(train_x),batch_size=BATCH_SIZE,shuffle=True)
    torch.save({'tensors':tensors,'factors':factors,'patterns':patterns,'config':{'seed':SEED,'samples':N_SAMPLES,'rows':ROWS,'cols':COLS,'noise_std':NOISE_STD}},OUTPUT_DIR/'dataset.pt')
    model=TinyAutoencoder(ROWS*COLS,LATENT_DIM).to(device); opt=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY); loss_fn=nn.MSELoss()
    best_test=float('inf'); best_epoch=0; best_state=None; history={'epoch':[],'train_mse':[],'test_mse':[]}; trajectory=[]; start=time.time()
    for epoch in range(1,EPOCHS+1):
        model.train()
        for (batch,) in loader:
            batch=batch.to(device); opt.zero_grad(set_to_none=True); recon,_=model(batch); loss=loss_fn(recon,batch); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            tr,_=model(train_x.to(device)); te,z=model(test_x.to(device)); train_mse=normalized_mse(tr,train_x.to(device)); test_mse=normalized_mse(te,test_x.to(device)); mean_z=z.mean(0).cpu().tolist()
        history['epoch'].append(epoch); history['train_mse'].append(train_mse); history['test_mse'].append(test_mse); trajectory.append({'epoch':epoch,'test_mse':test_mse,'latent_mean':mean_z})
        if test_mse<best_test: best_test=test_mse; best_epoch=epoch; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if epoch==1 or epoch%50==0:
            print(f'epoch {epoch:4d}/{EPOCHS} | normalized MSE {test_mse:.8f} | mean latent=[{mean_z[0]: .5f}, {mean_z[1]: .5f}] | time={time.time()-start:.1f}s')
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad(): tr,train_latent=model(train_x.to(device)); te,test_latent=model(test_x.to(device))
    train_latent=train_latent.cpu().numpy(); test_latent=test_latent.cpu().numpy(); train_mse=normalized_mse(tr.cpu(),train_x); test_mse=normalized_mse(te.cpu(),test_x)
    train_corr=correlation_matrix(train_latent,train_factors.numpy()); test_corr=correlation_matrix(test_latent,test_factors.numpy())
    torch.save({'model_state_dict':best_state,'best_epoch':best_epoch,'best_test_mse':best_test,'config':{'rows':ROWS,'cols':COLS,'latent_dim':LATENT_DIM}},OUTPUT_DIR/'best_model.pt')
    save_training_curve(history,OUTPUT_DIR/'training_curve.png'); save_latent_scatter(test_latent,test_factors.numpy(),OUTPUT_DIR/'latent_scatter.png')
    (OUTPUT_DIR/'latent_trajectory.json').write_text(json.dumps(trajectory,indent=2))
    results={'experiment':'known_factor_synthetic_latent_v1','seed':SEED,'samples':N_SAMPLES,'tensor_shape':[ROWS,COLS],'latent_dim':LATENT_DIM,'noise_std':NOISE_STD,'epochs':EPOCHS,'best_epoch':best_epoch,'best_test_normalized_mse':best_test,'final_train_normalized_mse':train_mse,'final_test_normalized_mse':test_mse,'train_latent_factor_correlations':train_corr,'test_latent_factor_correlations':test_corr,'interpretation_note':'Latent coordinates can rotate, reflect, scale, or nonlinearly mix factors; direct one-to-one alignment is not required.'}
    (OUTPUT_DIR/'results.json').write_text(json.dumps(results,indent=2))
    (OUTPUT_DIR/'README.txt').write_text('Known factors: factor_A, factor_B, factor_C, interaction_strength. Generation: X=A*a+B*b+C*c+D*(a*b*d)+noise. The model never receives the factors. Compare learned latent with the saved factors.\n')
    print('\n'+'='*80); print('DONE'); print('='*80); print(f'Best epoch: {best_epoch}'); print(f'Best test MSE: {best_test:.10f}'); print(f'Results directory: {OUTPUT_DIR.resolve()}'); print('\nTEST LATENT <-> KNOWN FACTOR CORRELATIONS')
    for z,vals in test_corr.items():
        print(z); [print(f'    {k:22s}: {v:+.5f}') for k,v in vals.items()]

if __name__=='__main__': main()
