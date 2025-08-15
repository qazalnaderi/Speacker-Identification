import os, glob, random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SpeakerDataset, collate_fn
from tdnn import XVectorTDNN
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

MAX_TRAIN_FILES = 1000
MAX_VAL_FILES = 700
FEAT_DIM = 39

ROOT_DEV  = "raw_data/vox1_dev_wav/wav"
ROOT_TEST = "raw_data/vox1_test_wav/wav"


speakers_dev = sorted([d for d in os.listdir(ROOT_DEV) if os.path.isdir(os.path.join(ROOT_DEV, d))])
spk2idx      = {spk:i for i, spk in enumerate(speakers_dev)}
NUM_SPK      = len(spk2idx)

def split_speakers(speakers, train_ratio=0.85, seed=123):
    r = random.Random(seed)
    lst = speakers[:]
    r.shuffle(lst)
    n_train = int(len(lst) * train_ratio)
    return set(lst[:n_train]), set(lst[n_train:])

def files_for_speakers(ROOT, spk_set, map_spk=True):
    items = []
    for spk in spk_set:
        for f in glob.glob(os.path.join(ROOT, spk, "**", "*.wav"), recursive=True):
            items.append((f, spk2idx[spk] if map_spk else spk))
    return items

train_spks, val_spks = split_speakers(speakers_dev)
train_list = files_for_speakers(ROOT_DEV, train_spks, map_spk=True)[:MAX_TRAIN_FILES]
val_list   = files_for_speakers(ROOT_DEV, val_spks,   map_spk=True)[:MAX_VAL_FILES]

train_ds = SpeakerDataset(train_list, augment_audio=True)
val_ds   = SpeakerDataset(val_list,   augment_audio=False)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = XVectorTDNN(feat_dim=FEAT_DIM, n_spk=NUM_SPK).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# ===== Training =====
print("Training started")
for epoch in range(5):
    model.train()
    total_loss, total_correct = 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits, _ = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        total_correct += (logits.argmax(dim=1) == yb).sum().item()
    print(f"[Epoch {epoch+1}] Loss: {total_loss/len(train_ds):.4f} | Acc: {total_correct/len(train_ds):.4f}")

# ===== Clustering =====
model.eval()
embeddings, labels = [], []
with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(device)
        _, emb = model(xb)
        embeddings.append(emb.cpu().numpy())
        labels.append(yb.numpy())

embeddings = np.vstack(embeddings)
labels = np.hstack(labels)

kmeans = KMeans(n_clusters=2, random_state=0).fit(embeddings)
pca = PCA(n_components=2).fit_transform(embeddings)

plt.scatter(pca[:,0], pca[:,1], c=kmeans.labels_, cmap="coolwarm", s=30)
plt.title("Speaker Clustering")
plt.show()
