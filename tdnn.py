import os, glob, random
import torch
import torch.nn as nn

ROOT_DEV  = "raw_data/vox1_dev_wav/wav"

speakers_dev = sorted([d for d in os.listdir(ROOT_DEV) if os.path.isdir(os.path.join(ROOT_DEV, d))])
spk2idx      = {spk:i for i, spk in enumerate(speakers_dev)}
idx2spk      = {i:spk for spk, i in spk2idx.items()}

FEAT_DIM = 39
NUM_SPK  = len(spk2idx)

class TDNNLayer(nn.Module):
    def __init__(self, in_ch, out_ch, k=5, d=1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, dilation=d)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.act  = nn.ReLU()
    def forward(self, x):  # (B, F, T)
        return self.act(self.bn(self.conv(x)))

class XVectorTDNN(nn.Module):
    def __init__(self, feat_dim=FEAT_DIM, n_spk=NUM_SPK):
        super().__init__()
        self.t1 = TDNNLayer(feat_dim, 512, k=5, d=1)
        self.t2 = TDNNLayer(512,      512, k=3, d=2)
        self.t3 = TDNNLayer(512,      512, k=3, d=3)
        self.t4 = TDNNLayer(512,      512, k=1, d=1)
        self.t5 = TDNNLayer(512,      1500, k=1, d=1)
        self.fc1 = nn.Linear(1500*2, 512)
        self.fc2 = nn.Linear(512, 256)      # embedding
        self.cls = nn.Linear(256, n_spk)

    def stats_pool(self, h):  # (B, C, T)
        mu  = h.mean(dim=2)
        std = h.std(dim=2)
        return torch.cat([mu, std], dim=1)  # (B, 2C)

    def forward(self, x):     # (B, T, F)
        x = x.transpose(1, 2) # → (B, F, T)
        h = self.t5(self.t4(self.t3(self.t2(self.t1(x)))))
        z = self.stats_pool(h)              # (B, 3000)
        z = torch.relu(self.fc1(z))
        emb = self.fc2(z)                   # (B, 256)
        logits = self.cls(emb)              # (B, n_spk)
        return logits, emb
