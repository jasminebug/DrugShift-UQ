"""Experimental cross-gated, low-rank cross-modal fusion.

Hypothesis: reciprocal gates + factorized interactions help generalize across
previously unseen drugs/cells. Original configuration, NOT a verified claim
of first-in-literature novelty or predictive superiority.
"""
from copy import deepcopy
import os
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, inp, dim=64):
        super().__init__()
        self.seq=nn.Sequential(nn.Linear(inp,128),nn.LayerNorm(128),nn.GELU(),
                               nn.Dropout(.1),nn.Linear(128,dim),nn.GELU())
    def forward(self,x):return self.seq(x)

class CrossGatedInteraction(nn.Module):
    """Reciprocal cross-modal gates, factorized multiplicative interaction + residual fusion."""
    def __init__(self,nc,nd,variant='full',dim=64,rank=32):
        super().__init__();self.variant=variant
        if variant=='early_fusion':
            self.mlp=nn.Sequential(nn.Linear(nc+nd,192),nn.LayerNorm(192),nn.GELU(),
                      nn.Dropout(.15),nn.Linear(192,96),nn.GELU(),nn.Linear(96,1))
            return
        self.c=Encoder(nc,dim);self.d=Encoder(nd,dim)
        self.gate_c=nn.Linear(dim,dim);self.gate_d=nn.Linear(dim,dim)
        self.U=nn.Linear(dim,rank,bias=False);self.V=nn.Linear(dim,rank,bias=False)
        self.head=nn.Sequential(nn.Linear(dim*4+rank,128),nn.LayerNorm(128),nn.GELU(),
                    nn.Dropout(.15),nn.Linear(128,64),nn.GELU(),nn.Linear(64,1))
    def forward(self,cx,dx):
        if self.variant=='early_fusion':return self.mlp(torch.cat([cx,dx],dim=-1)).squeeze(-1)
        c=self.c(cx);d=self.d(dx)
        if self.variant!='no_gate':
            # residual cross-gating avoids suppressing the original encoder signal.
            cg=c*(1.0+torch.sigmoid(self.gate_c(d)))
            dg=d*(1.0+torch.sigmoid(self.gate_d(c)))
        else:cg,dg=c,d
        bilinear=(self.U(cg)*self.V(dg)) if self.variant!='no_bilinear' else torch.zeros(
            (len(c),self.U.out_features),device=c.device,dtype=c.dtype)
        z=torch.cat([cg,dg,cg*dg,torch.abs(cg-dg),bilinear],dim=-1)
        return self.head(z).squeeze(-1)


def _save_atomic(obj,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    torch.save(obj,tmp)
    os.replace(tmp,path)


def _infer_device(device):
    if device=='auto':return 'cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    return device


def fit_member(ft,y,train,val,checkpoint,variant='full',seed=42,epochs=24,patience=6,
               batch_size=1024,lr=0.001,device='auto',config_tag=''):
    device=_infer_device(device)
    torch.set_num_threads(max(1,min(4,os.cpu_count() or 2)))
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    if device=='cuda':torch.cuda.manual_seed_all(seed)
    net=CrossGatedInteraction(ft.cv.shape[1],ft.dv.shape[1],variant=variant).to(device)
    optimizer=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=1e-4)
    yr=y[train];ym=float(np.mean(yr));ys=max(float(np.std(yr)),1e-6)
    c=torch.from_numpy(ft.cv).to(device);d=torch.from_numpy(ft.dv).to(device)
    ci=torch.from_numpy(ft.cids.astype(np.int64)).to(device)
    di=torch.from_numpy(ft.dids.astype(np.int64)).to(device)
    yt=torch.from_numpy(((y-ym)/ys).astype('float32')).to(device)
    random_gen=np.random.default_rng(seed)
    best_loss=float('inf');best_state=None;stale=0;start=0;history=[]
    if Path(checkpoint).exists():
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if ck.get('config_tag')!=config_tag:
            raise RuntimeError('Old checkpoint configuration differs; use another run directory or clear safely.')
        net.load_state_dict(ck['net']);optimizer.load_state_dict(ck['optimizer'])
        # Optimizer tensors must follow the chosen training device (CPU/GPU).
        for state in optimizer.state.values():
            for k,v in state.items():
                if isinstance(v,torch.Tensor):state[k]=v.to(device)
        start=int(ck['epoch']);best_loss=float(ck['best_loss']);stale=int(ck['stale'])
        best_state=ck['best_state'];history=ck['history'];random_gen.bit_generator.state=ck['rng_numpy']
        torch.set_rng_state(ck['rng_torch'])
        if device=='cuda' and ck.get('rng_cuda') is not None:torch.cuda.set_rng_state_all(ck['rng_cuda'])
        print(f'[checkpoint] {Path(checkpoint).name}: resuming after completed epoch {start}',flush=True)
    for epoch in range(start,epochs):
        if stale>=patience:break
        net.train();total=0.0;count=0
        ordered=random_gen.permutation(train)
        for s in range(0,len(ordered),batch_size):
            rows=torch.as_tensor(ordered[s:s+batch_size],dtype=torch.long,device=device)
            output=net(c[ci[rows]],d[di[rows]])
            loss=torch.nn.functional.mse_loss(output,yt[rows])
            optimizer.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),max_norm=3.0);optimizer.step()
            total+=float(loss.item())*len(rows);count+=len(rows)
        net.eval();val_loss_sum=0.;val_count=0
        with torch.inference_mode():
            for s in range(0,len(val),batch_size*2):
                rows=torch.as_tensor(val[s:s+batch_size*2],dtype=torch.long,device=device)
                loss=torch.nn.functional.mse_loss(net(c[ci[rows]],d[di[rows]]),yt[rows],reduction='sum')
                val_loss_sum+=float(loss.item());val_count+=len(rows)
        vl=val_loss_sum/max(val_count,1)
        history.append({'epoch':epoch+1,'train_mse_scaled':total/max(count,1),'validation_mse_scaled':vl})
        if vl < best_loss-1e-5:
            best_loss=vl;best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};stale=0
        else:stale+=1
        _save_atomic({'config_tag':config_tag,'epoch':epoch+1,'net':net.state_dict(),
                      'optimizer':optimizer.state_dict(),'best_state':best_state,'best_loss':best_loss,
                      'stale':stale,'history':history,'rng_numpy':random_gen.bit_generator.state,
                      'rng_torch':torch.get_rng_state(),
                      'rng_cuda':torch.cuda.get_rng_state_all() if device=='cuda' else None},checkpoint)
        print(f'[epoch] {variant} seed={seed} {epoch+1}/{epochs} val_mse={vl:.5f} best={best_loss:.5f}',flush=True)
    if best_state is None:raise RuntimeError('No usable model weights from training')
    net.load_state_dict(best_state);net.eval()
    return {'model':net,'device':device,'ym':ym,'ys':ys,'history':history,
            'trained_epochs':len(history),'best_validation_mse':best_loss}


def predict(member,ft,rows,batch_size=4096):
    device=member['device'];net=member['model'];out=[]
    c=torch.from_numpy(ft.cv).to(device);d=torch.from_numpy(ft.dv).to(device)
    with torch.inference_mode():
        for start in range(0,len(rows),batch_size):
            idx=rows[start:start+batch_size]
            ci=torch.as_tensor(ft.cids[idx],dtype=torch.long,device=device)
            di=torch.as_tensor(ft.dids[idx],dtype=torch.long,device=device)
            out.append(net(c[ci],d[di]).detach().cpu().numpy())
    if not out:return np.array([],dtype='float32')
    return np.concatenate(out)*member['ys']+member['ym']
