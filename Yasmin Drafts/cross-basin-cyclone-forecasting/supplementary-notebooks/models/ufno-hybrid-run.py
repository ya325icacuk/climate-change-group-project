import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from pathlib import Path
from collections import Counter
from copy import deepcopy
from tqdm.auto import tqdm
import warnings, os
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.1)

PROJECT_ROOT = Path("../..").resolve()
DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR  = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

N_MODES = 12; HIDDEN_CH = 32; N_LAYERS = 3; PADDING = 9
IN_CHANNELS = 15; TIME_DIM = 6; TIME_EMB_DIM = 64
BATCH_SIZE = 64; LR = 5e-4; WEIGHT_DECAY = 1e-3
EPOCHS = 150; PATIENCE = 30; DIR_WEIGHT = 0.5
USE_ENV = True; USE_1D = True
FT_LR = 1e-4; FT_EPOCHS = 50; FT_PATIENCE = 15
N_DIR_CLASSES = 8; N_INT_CLASSES = 4
DIR_LABELS = ["E","SE","S","SW","W","NW","N","NE"]
INTE_LABELS = ["Weakening","Steady","Slow-intens.","Rapid-intens."]

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

SPLITS = {"wp_train":{"reflected":False}, "wp_val":{"reflected":False},
          "sp_test":{"reflected":True}, "sp_ft_train":{"reflected":True}, "sp_ft_val":{"reflected":True}}
raw = {}
for s in SPLITS:
    raw[s] = {k: torch.load(DATA_DIR/d/f"{s}_{f}.pt", weights_only=False)
              for k,d,f in [("grids","grids","grids"),("env","env","env"),
                            ("data1d","data1d","1d"),("labels","labels","labels"),("time","time","time")]}

class CycloneDataset(Dataset):
    def __init__(self, grids, env, d1d, labels, time, use_ref=False, d1d_mean=None, d1d_std=None):
        self.samples = []
        dk = "direction_reflected" if use_ref else "direction"
        for sid in grids:
            g,e,d,t,dl,il = grids[sid],env[sid],d1d[sid],time[sid],labels[sid][dk],labels[sid]["intensity"]
            for i in range(g.shape[0]):
                if dl[i].item()==-1 or il[i].item()==-1: continue
                self.samples.append((g[i],e[i],d[i],t[i],dl[i].long(),il[i].long()))
        if d1d_mean is None:
            a = torch.stack([s[2] for s in self.samples])
            self.d1d_mean, self.d1d_std = a.mean(0), a.std(0).clamp(min=1e-6)
        else: self.d1d_mean, self.d1d_std = d1d_mean, d1d_std
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        g,e,d,t,dl,il = self.samples[i]
        return g, e, (d-self.d1d_mean)/self.d1d_std, t, dl, il

datasets = {}
datasets["wp_train"] = CycloneDataset(raw["wp_train"]["grids"],raw["wp_train"]["env"],
    raw["wp_train"]["data1d"],raw["wp_train"]["labels"],raw["wp_train"]["time"])
dm,ds = datasets["wp_train"].d1d_mean, datasets["wp_train"].d1d_std
for s,c in SPLITS.items():
    if s=="wp_train": continue
    datasets[s] = CycloneDataset(raw[s]["grids"],raw[s]["env"],raw[s]["data1d"],
        raw[s]["labels"],raw[s]["time"],use_ref=c["reflected"],d1d_mean=dm,d1d_std=ds)
loaders = {s: DataLoader(datasets[s],batch_size=BATCH_SIZE,shuffle=(s=="wp_train"),
                         num_workers=0,pin_memory=True) for s in SPLITS}
for s,d in datasets.items(): print(f"  {s:15s}: {len(d):5d}")

# Class weights
dc,ic = Counter(),Counter()
for *_,dl,il in datasets["wp_train"].samples: dc[dl.item()]+=1; ic[il.item()]+=1
n = len(datasets["wp_train"])
dir_weights = torch.tensor([n/(N_DIR_CLASSES*max(dc[c],1)) for c in range(N_DIR_CLASSES)])
int_weights = torch.tensor([n/(N_INT_CLASSES*max(ic[c],1)) for c in range(N_INT_CLASSES)])
print("Dir:", dir_weights.numpy().round(3))
print("Int:", int_weights.numpy().round(3))

class SpectralConv2d(nn.Module):
    """2D Fourier spectral conv with reflect padding."""
    def __init__(self, in_ch, out_ch, modes1, modes2, padding=9):
        super().__init__()
        self.modes1, self.modes2, self.out_channels, self.padding = modes1, modes2, out_ch, padding
        s = (2/(in_ch+out_ch))**0.5
        self.w1 = nn.Parameter(s*(torch.rand(in_ch,out_ch,modes1,modes2,dtype=torch.cfloat)-0.5))
        self.w2 = nn.Parameter(s*(torch.rand(in_ch,out_ch,modes1,modes2,dtype=torch.cfloat)-0.5))

    def forward(self, x):
        B,C,H,W = x.shape
        if self.padding > 0:
            x = F.pad(x, [self.padding]*4, mode='reflect')
        Hp,Wp = x.shape[-2], x.shape[-1]
        xf = torch.fft.rfft2(x)
        of = torch.zeros(B,self.out_channels,Hp,Wp//2+1,dtype=torch.cfloat,device=x.device)
        of[:,:,:self.modes1,:self.modes2] = torch.einsum("bixy,ioxy->boxy",xf[:,:,:self.modes1,:self.modes2],self.w1)
        of[:,:,-self.modes1:,:self.modes2] = torch.einsum("bixy,ioxy->boxy",xf[:,:,-self.modes1:,:self.modes2],self.w2)
        x = torch.fft.irfft2(of, s=(Hp,Wp))
        if self.padding > 0:
            x = x[:,:,self.padding:-self.padding,self.padding:-self.padding]
        return x


class UNetBranch(nn.Module):
    """Lightweight single-level U-Net branch for local features."""
    def __init__(self, ch):
        super().__init__()
        self.down = nn.Sequential(nn.Conv2d(ch,ch,3,stride=2,padding=1,bias=False), nn.BatchNorm2d(ch), nn.GELU())
        self.mid = nn.Sequential(nn.Conv2d(ch,ch,3,padding=1,bias=False), nn.BatchNorm2d(ch), nn.GELU())
        self.up = nn.ConvTranspose2d(ch,ch,2,stride=2)
        self.fuse = nn.Sequential(nn.Conv2d(ch*2,ch,1,bias=False), nn.BatchNorm2d(ch))

    def forward(self, x):
        skip = x
        x = self.down(x)
        x = self.mid(x)
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.fuse(torch.cat([x, skip], 1))


class FiLMLayer(nn.Module):
    def __init__(self, cond_dim, ch):
        super().__init__()
        self.fc = nn.Linear(cond_dim, ch*2)
        nn.init.zeros_(self.fc.weight); nn.init.zeros_(self.fc.bias)
        self.fc.bias.data[:ch] = 1.0
    def forward(self, x, c):
        g,b = self.fc(c).chunk(2,1)
        return g.unsqueeze(-1).unsqueeze(-1)*x + b.unsqueeze(-1).unsqueeze(-1)


class UFNOBlock(nn.Module):
    """U-FNO block: 3 gated branches (spectral + unet + residual) + FiLM."""
    def __init__(self, ch, modes, padding, time_emb_dim, dropout=0.1):
        super().__init__()
        self.spectral = SpectralConv2d(ch, ch, modes, modes, padding)
        self.unet = UNetBranch(ch)
        self.residual = nn.Conv2d(ch, ch, 1)
        self.gate = nn.Parameter(torch.ones(3)/3)
        self.norm = nn.BatchNorm2d(ch)
        self.film = FiLMLayer(time_emb_dim, ch)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x, t_emb):
        g = F.softmax(self.gate, dim=0)
        out = g[0]*self.spectral(x) + g[1]*self.unet(x) + g[2]*self.residual(x)
        out = self.norm(out)
        if t_emb is not None:
            out = self.film(out, t_emb)
        out = self.dropout(F.gelu(out))
        return out + x

class UFNO2dClassifier(nn.Module):
    """U-FNO with FiLM time conditioning."""
    def __init__(self, in_ch=15, hidden_ch=32, n_modes=12, n_layers=3, padding=9,
                 n_dir=8, n_int=4, env_dim=40, d1d_dim=4,
                 use_env=True, use_1d=True, time_dim=6, time_emb_dim=64, dropout=0.1):
        super().__init__()
        self.use_env, self.use_1d = use_env, use_1d
        self.time_mlp = nn.Sequential(nn.Linear(time_dim,time_emb_dim),nn.GELU(),nn.Linear(time_emb_dim,time_emb_dim))
        self.lifting = nn.Sequential(nn.Conv2d(in_ch,hidden_ch,1),nn.GELU())
        self.blocks = nn.ModuleList([UFNOBlock(hidden_ch,n_modes,padding,time_emb_dim,dropout) for _ in range(n_layers)])
        self.projection = nn.Sequential(nn.Conv2d(hidden_ch,hidden_ch,1),nn.GELU())
        aux = (env_dim if use_env else 0) + (d1d_dim if use_1d else 0)
        h_in = hidden_ch + aux
        self.head_dir = nn.Sequential(nn.Linear(h_in,128),nn.GELU(),nn.Dropout(0.2),
                                       nn.Linear(128,64),nn.GELU(),nn.Dropout(0.1),nn.Linear(64,n_dir))
        self.head_int = nn.Sequential(nn.Linear(h_in,128),nn.GELU(),nn.Dropout(0.2),
                                       nn.Linear(128,64),nn.GELU(),nn.Dropout(0.1),nn.Linear(64,n_int))

    def forward(self, grid, env=None, d1d=None, time_feat=None):
        t = self.time_mlp(time_feat) if time_feat is not None else None
        x = self.lifting(grid)
        for blk in self.blocks: x = blk(x, t)
        x = self.projection(x).mean(dim=(-2,-1))
        parts = [x]
        if self.use_env and env is not None: parts.append(env)
        if self.use_1d and d1d is not None: parts.append(d1d)
        x = torch.cat(parts, -1)
        return self.head_dir(x), self.head_int(x)

model = UFNO2dClassifier(hidden_ch=HIDDEN_CH, n_modes=N_MODES, n_layers=N_LAYERS,
                          padding=PADDING, time_dim=TIME_DIM, time_emb_dim=TIME_EMB_DIM).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"U-FNO: {n_params:,} params")
with torch.no_grad():
    d,i = model(torch.randn(2,15,81,81,device=DEVICE), torch.randn(2,40,device=DEVICE),
                torch.randn(2,4,device=DEVICE), torch.randn(2,6,device=DEVICE))
    print(f"Output: dir={d.shape}, int={i.shape}")

loss_dir_fn = nn.CrossEntropyLoss(weight=dir_weights.to(DEVICE), label_smoothing=0.05)
loss_int_fn = nn.CrossEntropyLoss(weight=int_weights.to(DEVICE), label_smoothing=0.05)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    tl=cd=ci=tot=0
    for g,e,d,t,dl,il in loader:
        g,e,d,t,dl,il = [x.to(device) for x in [g,e,d,t,dl,il]]
        do,io = model(g,e,d,t)
        loss = DIR_WEIGHT*loss_dir_fn(do,dl)+(1-DIR_WEIGHT)*loss_int_fn(io,il)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
        bs=g.size(0); tl+=loss.item()*bs; cd+=(do.argmax(1)==dl).sum().item()
        ci+=(io.argmax(1)==il).sum().item(); tot+=bs
    return tl/tot, cd/tot, ci/tot

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tl=tot=0; dp,dt,ip,it_=[],[],[],[]
    for g,e,d,t,dl,il in loader:
        g,e,d,t,dl,il = [x.to(device) for x in [g,e,d,t,dl,il]]
        do,io = model(g,e,d,t)
        loss = DIR_WEIGHT*loss_dir_fn(do,dl)+(1-DIR_WEIGHT)*loss_int_fn(io,il)
        bs=g.size(0); tl+=loss.item()*bs; tot+=bs
        dp.extend(do.argmax(1).cpu().tolist()); dt.extend(dl.cpu().tolist())
        ip.extend(io.argmax(1).cpu().tolist()); it_.extend(il.cpu().tolist())
    return {"loss":tl/tot, "dir_acc":accuracy_score(dt,dp), "int_acc":accuracy_score(it_,ip),
            "dir_f1":f1_score(dt,dp,average="macro",zero_division=0),
            "int_f1":f1_score(it_,ip,average="macro",zero_division=0),
            "dir_pred":dp,"dir_true":dt,"int_pred":ip,"int_true":it_}

history = {k:[] for k in ["train_loss","val_loss","train_dir_acc","val_dir_acc","train_int_acc","val_int_acc"]}
best_val_loss = float("inf"); best_state = None; pat = 0

for epoch in range(1, EPOCHS+1):
    tl,tda,tia = train_one_epoch(model, loaders["wp_train"], optimizer, DEVICE)
    vm = evaluate(model, loaders["wp_val"], DEVICE)
    scheduler.step()
    for k,v in [("train_loss",tl),("val_loss",vm["loss"]),("train_dir_acc",tda),
                ("val_dir_acc",vm["dir_acc"]),("train_int_acc",tia),("val_int_acc",vm["int_acc"])]: history[k].append(v)
    if vm["loss"]<best_val_loss:
        best_val_loss=vm["loss"]; best_state=deepcopy(model.state_dict()); pat=0; mk=" *"
    else: pat+=1; mk=""
    if epoch%5==0 or epoch==1 or mk:
        print(f"Ep {epoch:3d}/{EPOCHS} | T loss={tl:.4f} dir={tda:.3f} int={tia:.3f} | "
              f"V loss={vm['loss']:.4f} dir={vm['dir_acc']:.3f} int={vm['int_acc']:.3f}{mk}")
    if pat>=PATIENCE: print(f"\nEarly stop at ep {epoch}"); break

model.load_state_dict(best_state)
print(f"\nBest val loss: {best_val_loss:.4f}, dir acc: {max(history['val_dir_acc']):.4f}")

fig, axes = plt.subplots(1,3,figsize=(15,4))
for ax,(t,v,title) in zip(axes,[("train_loss","val_loss","Loss"),("train_dir_acc","val_dir_acc","Dir Acc"),("train_int_acc","val_int_acc","Int Acc")]):
    ax.plot(history[t],label="Train"); ax.plot(history[v],label="Val")
    ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend()
fig.suptitle("U-FNO Training Curves",fontweight="bold"); fig.tight_layout()
fig.savefig(FIG_DIR/"ufno_training_curves.png",dpi=150,bbox_inches="tight")
plt.show()

def plot_cm(metrics, title):
    fig, axes = plt.subplots(1,2,figsize=(12,5))
    for ax,key,labels,cmap in [(axes[0],"dir",DIR_LABELS,"Blues"),(axes[1],"int",INTE_LABELS,"Oranges")]:
        cm = confusion_matrix(metrics[f"{key}_true"],metrics[f"{key}_pred"],labels=range(len(labels)))
        cm_pct = cm/cm.sum(axis=1,keepdims=True)*100
        sns.heatmap(cm_pct,annot=True,fmt=".0f",cmap=cmap,xticklabels=labels,yticklabels=labels,
                    cbar=False,ax=ax,vmin=0,vmax=100)
        ax.set_title(f"{title} — {key.title()} ({metrics[f'{key}_acc']:.1%})")
        ax.set_xlabel("Predicted")
    fig.tight_layout(); return fig

wp_m = evaluate(model, loaders["wp_val"], DEVICE)
print(f"WP Val — dir={wp_m['dir_acc']:.3f} F1={wp_m['dir_f1']:.3f} | int={wp_m['int_acc']:.3f}")
fig = plot_cm(wp_m, "U-FNO WP Val")
fig.savefig(FIG_DIR/"ufno_cm_wp.png",dpi=150,bbox_inches="tight"); plt.show()

sp_zs = evaluate(model, loaders["sp_test"], DEVICE)
print(f"SP Zero-Shot — dir={sp_zs['dir_acc']:.3f} | int={sp_zs['int_acc']:.3f}")
print(f"Transfer gap (dir): {sp_zs['dir_acc']-wp_m['dir_acc']:+.3f}")
fig = plot_cm(sp_zs, "U-FNO SP Zero-Shot")
fig.savefig(FIG_DIR/"ufno_cm_sp_zs.png",dpi=150,bbox_inches="tight"); plt.show()

def finetune(state, ftl, fvl, freeze=False, lr=FT_LR, eps=FT_EPOCHS, pat=FT_PATIENCE):
    m = UFNO2dClassifier(hidden_ch=HIDDEN_CH,n_modes=N_MODES,n_layers=N_LAYERS,
                          padding=PADDING,time_dim=TIME_DIM,time_emb_dim=TIME_EMB_DIM).to(DEVICE)
    m.load_state_dict(state)
    if freeze:
        for n,p in m.named_parameters():
            if "head" not in n: p.requires_grad=False
    opt = torch.optim.AdamW(filter(lambda p:p.requires_grad,m.parameters()),lr=lr,weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=eps,eta_min=1e-6)
    ba,bs_,w = 0.0,None,0
    for ep in range(1,eps+1):
        m.train()
        for g,e,d,t,dl,il in ftl:
            g,e,d,t,dl,il=[x.to(DEVICE) for x in [g,e,d,t,dl,il]]
            do,io=m(g,e,d,t)
            loss=DIR_WEIGHT*loss_dir_fn(do,dl)+(1-DIR_WEIGHT)*loss_int_fn(io,il)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
        sch.step()
        vm=evaluate(m,fvl,DEVICE)
        if vm["dir_acc"]>ba: ba=vm["dir_acc"];bs_=deepcopy(m.state_dict());w=0
        else: w+=1
        if ep%10==0: print(f"  FT {ep:3d} | dir={vm['dir_acc']:.3f} int={vm['int_acc']:.3f}")
        if w>=pat: print(f"  Stop at {ep}"); break
    m.load_state_dict(bs_); return m,ba

print("Full FT:")
ft_full,ft_full_a = finetune(best_state,loaders["sp_ft_train"],loaders["sp_ft_val"])
print(f"  Best: {ft_full_a:.3f}")
print("Head FT:")
ft_head,ft_head_a = finetune(best_state,loaders["sp_ft_train"],loaders["sp_ft_val"],freeze=True)
print(f"  Best: {ft_head_a:.3f}")

ft_best = ft_full if ft_full_a>=ft_head_a else ft_head
sp_ft = evaluate(ft_best, loaders["sp_test"], DEVICE)
print(f"SP Fine-Tuned — dir={sp_ft['dir_acc']:.3f} F1={sp_ft['dir_f1']:.3f} | int={sp_ft['int_acc']:.3f}")
fig = plot_cm(sp_ft, "U-FNO SP Fine-Tuned")
fig.savefig(FIG_DIR/"ufno_cm_sp_ft.png",dpi=150,bbox_inches="tight"); plt.show()

# Save checkpoints
torch.save(best_state, PROJECT_ROOT/"checkpoints"/"ufno_best_wp.pt")
torch.save(ft_best.state_dict(), PROJECT_ROOT/"checkpoints"/"ufno_best_ft.pt")
print("Checkpoints saved.")

# ── Gate values per layer ──
gate_vals = []
for i, blk in enumerate(model.blocks):
    g = F.softmax(blk.gate.detach().cpu(), dim=0)
    gate_vals.append(g.numpy())
    print(f"Layer {i+1}: Spectral={g[0]:.3f}, U-Net={g[1]:.3f}, Residual={g[2]:.3f}")

gate_vals = np.array(gate_vals)
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(1, N_LAYERS+1)
w = 0.25
ax.bar(x-w, gate_vals[:,0], w, label="Spectral", color="#2196F3")
ax.bar(x,   gate_vals[:,1], w, label="U-Net",    color="#4CAF50")
ax.bar(x+w, gate_vals[:,2], w, label="Residual", color="#FF9800")
ax.set_xlabel("Layer"); ax.set_ylabel("Gate Weight")
ax.set_title("U-FNO Learned Gate Weights per Layer")
ax.set_xticks(x); ax.legend(); ax.set_ylim(0,1)
fig.tight_layout()
fig.savefig(FIG_DIR/"ufno_gate_weights.png",dpi=150,bbox_inches="tight")
plt.show()

print("\n" + "="*80)
print(" U-FNO — Final Results")
print("="*80)
print(f"{'Setting':20s} | {'Dir Acc':>8s} | {'Dir F1':>7s} | {'Int Acc':>8s} | {'Int F1':>7s}")
print("-"*80)
for name, m in [("WP Val",wp_m),("SP Zero-Shot",sp_zs),("SP Fine-Tuned",sp_ft)]:
    print(f"{name:20s} | {m['dir_acc']:>7.1%} | {m['dir_f1']:>6.1%} | {m['int_acc']:>7.1%} | {m['int_f1']:>6.1%}")
print("="*80)
print(f"Parameters: {n_params:,}")
print(f"Transfer gap (dir): {sp_zs['dir_acc']-wp_m['dir_acc']:+.3f}")
print(f"FT recovery (dir):  {sp_ft['dir_acc']-sp_zs['dir_acc']:+.3f}")
print(f"\nGate summary:")
for i, blk in enumerate(model.blocks):
    g = F.softmax(blk.gate.detach().cpu(), dim=0)
    print(f"  Layer {i+1}: Spectral={g[0]:.2f} U-Net={g[1]:.2f} Residual={g[2]:.2f}")
