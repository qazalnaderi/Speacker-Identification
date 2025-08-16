import os, glob, random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SpeakerDataset, collate_fn
from tdnn import XVectorTDNN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import librosa.display
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import defaultdict, Counter

MAX_TRAIN_FILES = 2000
MAX_VAL_FILES = 1000
FEAT_DIM = 39

ROOT_DEV  = "raw_data/vox1_dev_wav/wav"
ROOT_TEST = "raw_data/vox1_test_wav/wav"

test_spks = sorted([d for d in os.listdir(ROOT_TEST) if os.path.isdir(os.path.join(ROOT_TEST, d))])
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

test_spk2idx = {spk: i for i, spk in enumerate(test_spks)}
test_list_mapped = []
for spk in test_spks:
    for f in glob.glob(os.path.join(ROOT_TEST, spk, "**", "*.wav"), recursive=True):
        test_list_mapped.append((f, test_spk2idx[spk]))
test_list = test_list_mapped 

train_ds = SpeakerDataset(train_list, augment_audio=True)
val_ds   = SpeakerDataset(val_list, augment_audio=False)
test_ds  = SpeakerDataset(test_list, augment_audio=False)

test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = XVectorTDNN(feat_dim=FEAT_DIM, n_spk=NUM_SPK).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

print("Training started")
for epoch in range(3):
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

def extract_embeddings(model, dataloader, device):
    """Extract embeddings for all samples in dataloader"""
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            _, emb = model(xb)  
            embeddings.append(emb.cpu().numpy())
            if isinstance(yb, torch.Tensor):
                labels.extend(yb.cpu().numpy().tolist())
            else:
                labels.extend(yb)
    
    return np.vstack(embeddings), labels

def create_speaker_prototypes(embeddings, labels):
    """Create speaker prototypes by averaging embeddings"""
    prototypes = {}
    label_to_embeddings = defaultdict(list)
    
    for emb, label in zip(embeddings, labels):
        label_to_embeddings[label].append(emb)
    
    for label, emb_list in label_to_embeddings.items():
        prototypes[label] = np.mean(emb_list, axis=0)
    
    return prototypes

def identify_speakers_cosine(test_embeddings, train_prototypes, threshold=0.75):
    """Identify speakers using cosine similarity + Unknown handling"""
    predictions = []
    confidences = []
    
    train_speakers = list(train_prototypes.keys())
    train_embs = np.array([train_prototypes[spk] for spk in train_speakers])
    
    for test_emb in test_embeddings:
        similarities = cosine_similarity([test_emb], train_embs)[0]
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        if best_score < threshold:  
            predictions.append("Unknown")
        else:
            predictions.append(int(train_speakers[best_idx]))
        confidences.append(float(best_score))
    
    return predictions, confidences


def identify_speakers_euclidean(test_embeddings, train_prototypes):
    """Identify speakers using Euclidean distance"""
    predictions = []
    confidences = []
    
    train_speakers = list(train_prototypes.keys())
    train_embs = np.array([train_prototypes[spk] for spk in train_speakers])
    
    for test_emb in test_embeddings:
        distances = np.linalg.norm(train_embs - test_emb, axis=1)
        
        best_idx = np.argmin(distances)
        predictions.append(int(train_speakers[best_idx])) 
        confidences.append(float(1.0 / (1.0 + distances[best_idx])))  
    
    return predictions, confidences

train_embeddings, train_labels = extract_embeddings(model, train_loader, device)

train_prototypes = create_speaker_prototypes(train_embeddings, train_labels)

test_embeddings, test_speaker_ids = extract_embeddings(model, test_loader, device)

pred_cosine, conf_cosine = identify_speakers_cosine(test_embeddings, train_prototypes)

pred_euclidean, conf_euclidean = identify_speakers_euclidean(test_embeddings, train_prototypes)

idx2spk = {v: k for k, v in spk2idx.items()}
test_idx2spk = {v: k for k, v in test_spk2idx.items()}
pred_names_cosine = [idx2spk[p] for p in pred_cosine]
pred_names_euclidean = [idx2spk[p] for p in pred_euclidean]

y_true = [test_idx2spk[idx] for idx in test_speaker_ids]

y_pred = []
for p in pred_cosine:
    if p == "Unknown":
        y_pred.append("Unknown")
    else:
        y_pred.append(idx2spk[p])   

spk_predictions = defaultdict(list)

for i, test_idx in enumerate(test_speaker_ids):
    test_spk_name = test_idx2spk[test_idx] 
    spk_predictions[test_spk_name].append(pred_names_cosine[i])

print("\nFinal speaker-level predictions:")
for spk, preds in spk_predictions.items():
    most_common_pred, count = Counter(preds).most_common(1)[0]
    print(f"Test speaker {spk} → Predicted as {most_common_pred} ({count}/{len(preds)} files)")
correct = 0
total = 0

print("\nFinal speaker-level predictions with consistency:")
for spk, preds in spk_predictions.items():
    most_common_pred, count = Counter(preds).most_common(1)[0]
    consistency = count / len(preds)
    print(f"Test speaker {spk} → Predicted as {most_common_pred} "
          f"({count}/{len(preds)} files, consistency={consistency:.2f})")
    if consistency > 0.7: 
        correct += 1
    total += 1

print(f"\nConsistency-based accuracy: {correct}/{total} = {correct/total:.2f}")


print(f"Total unique speakers in training: {len(train_prototypes)}")
print(f"Total test samples: {len(test_embeddings)}")
print(f"Average confidence (Euclidean): {np.mean(conf_euclidean):.3f}")

# ===== GENDER CLUSTERING =====

meta = pd.read_csv("raw_data/vox1_vox1_meta.csv", sep='\t')
spk2gender = dict(zip(meta['VoxCeleb1 ID'], meta['Gender']))
gender_map = {'m': 0, 'f': 1}

print("Starting to do clustering...")
model.eval()
val_embeddings, val_labels = [], []
with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(device)
        _, emb = model(xb)
        val_embeddings.append(emb.cpu().numpy())
        val_labels.extend(yb.cpu().numpy().tolist())

val_embeddings = np.vstack(val_embeddings)
val_labels = np.array(val_labels)

y_gender_val = []
for spk_id in val_labels:
    spk_name = idx2spk[spk_id]
    try:
        gender = spk2gender[spk_name]
        y_gender_val.append(gender_map[gender])
    except KeyError:
        print(f"Warning: Speaker {spk_name} not found in metadata")
        y_gender_val.append(0)  

y_gender_val = np.array(y_gender_val)
print("Unique genders in validation:", np.unique(y_gender_val, return_counts=True))

test_idx2spk = {v: k for k, v in test_spk2idx.items()}
y_gender_test = []
for spk_idx in test_speaker_ids:
    spk_id = test_idx2spk[spk_idx] 
    try:
        gender = spk2gender[spk_id]
        y_gender_test.append(gender_map[gender])
    except KeyError:
        print(f"Warning: Test speaker {spk_id} not found in metadata")
        y_gender_test.append(0)  

y_gender_test = np.array(y_gender_test)
print("Unique genders in test:", np.unique(y_gender_test, return_counts=True))

print("Training...")
clf = LogisticRegression(max_iter=500)
clf.fit(val_embeddings, y_gender_val)

y_pred_gender = clf.predict(test_embeddings)

print("Report:")
acc = accuracy_score(y_gender_test, y_pred_gender)
precision = precision_score(y_gender_test, y_pred_gender)
recall    = recall_score(y_gender_test, y_pred_gender)
f1        = f1_score(y_gender_test, y_pred_gender)

print(f"Accuracy: {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")


pca = PCA(n_components=2)
points = pca.fit_transform(val_embeddings)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(points[:,0], points[:,1], c=y_gender_val, cmap='coolwarm', s=30, alpha=0.6)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Speaker Embeddings by Gender")
plt.colorbar(scatter, label="0 = Male, 1 = Female")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

import librosa
path, _ = val_list[0]
sig, sr = librosa.load(path, sr=16000)
stft = librosa.stft(sig, n_fft=2048, hop_length=512)
spectrogram = np.abs(stft)
log_spectrogram = librosa.amplitude_to_db(spectrogram)

plt.figure(figsize=(12, 6))
librosa.display.specshow(log_spectrogram, sr=sr, hop_length=512, x_axis='time', y_axis='log')
plt.colorbar(format="%+2.0f dB")
plt.title("Log-Spectrogram")
plt.tight_layout()
plt.show()
