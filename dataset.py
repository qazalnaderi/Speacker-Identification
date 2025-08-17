from torch.utils.data import Dataset, DataLoader
from preprocess import preprocess
import torch

class SpeakerDataset(Dataset):
    def __init__(self, file_list, augment_audio=False):
        self.file_list = file_list
        self.augment_audio = augment_audio
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, idx):
        path, label = self.file_list[idx]
        feats = preprocess(path, augment_audio=self.augment_audio)
        if feats is None:
            return None
        x = torch.from_numpy(feats)
        y = torch.tensor(label, dtype=torch.long)
        return x, y

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])