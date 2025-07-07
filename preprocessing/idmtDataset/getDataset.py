# getDataset.py

import os
from torch.utils.data import Dataset, DataLoader
import torchaudio
import pandas as pd
from import_idmt_traffic_dataset import import_idmt_traffic_dataset

class IDMTTrafficDataset(Dataset):
    """
    Custom Dataset for IDMT-Traffic audio.
    Excludes Bus class and merges background as 'no_vehicle'.
    """
    def __init__(self, annotation_txt_path, audio_root, transform=None):
        """
        Args:
            annotation_txt_path (str): Path to .txt listing WAV files.
            audio_root (str): Directory containing WAV audio files.
            transform (callable, optional): Optional transform on waveform.
        """
        # Load metadata from annotation file
        df = import_idmt_traffic_dataset(annotation_txt_path)
        # Exclude Bus samples
        df = df[df['vehicle'] != 'B'].reset_index(drop=True)
        # Assign label strings: background → 'no_vehicle', else vehicle code
        df['label_str'] = df.apply(
            lambda row: 'no_vehicle' if row['is_background'] else row['vehicle'],
            axis=1
        )
        # Map to integer labels
        self.label_map = {'C': 0, 'M': 1, 'T': 2, 'no_vehicle': 3}
        df['label'] = df['label_str'].map(self.label_map)
        # Store
        self.audio_root = audio_root
        self.data = df
        self.transform = transform
        print(f"Loaded IDMTTrafficDataset: {len(self.data)} samples, classes {list(self.label_map.keys())}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            waveform (Tensor): Audio signal.
            label (int): Class index.
        """
        file_stem = self.data.loc[idx, 'file']
        audio_path = os.path.join(self.audio_root, f"{file_stem}.wav")
        waveform, sample_rate = torchaudio.load(audio_path)
        if self.transform is not None:
            waveform = self.transform(waveform)
        label = int(self.data.loc[idx, 'label'])
        return waveform, label

def get_dataloader(annotation_txt_path,
                   audio_root,
                   batch_size=16,
                   shuffle=True,
                   num_workers=4,
                   transform=None):
    """
    Utility to create DataLoader for IDMT-Traffic four-class task.
    """
    dataset = IDMTTrafficDataset(annotation_txt_path, audio_root, transform=transform)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers)

# Example usage (in your training script):
# from getDataset import get_dataloader
# train_loader = get_dataloader(
#     annotation_txt_path='/path/to/eusipco_2021_train.txt',
#     audio_root='/path/to/audio',
#     batch_size=32,
#     shuffle=True
# )
