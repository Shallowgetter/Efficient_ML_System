import os
import pandas as pd
import torchaudio
from torch.utils.data import Dataset
from collections import Counter
import matplotlib.pyplot as plt

def parse_filename(filename):
    """
    Parse the filename to extract label (vehicle type).
    Merge left/right (L/R) direction and ignore multi-vehicle.
    Returns ('label', None) or (None, 'BG').
    """
    basename = os.path.splitext(filename)[0]
    if basename.endswith('-BG'):
        return None, 'BG'
    parts = basename.split('_')
    if len(parts) < 8:
        return None, None
    vehicle_part = parts[6]
    vehicle_type = vehicle_part[0].upper()
    if vehicle_type in ['B', 'C', 'M', 'T']:
        return vehicle_type, None
    else:
        return None, None

def plot_distribution_histogram(vehicle_counter, save_path='vehicle_distribution.png'):
    """
    Plot and save a histogram of vehicle type distribution.
    """
    print("\nGenerating vehicle type distribution histogram...")
    
    # Prepare data for plotting
    labels = list(vehicle_counter.keys())
    counts = list(vehicle_counter.values())
    
    # Create the histogram
    plt.figure(figsize=(12, 8))
    bars = plt.bar(labels, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
    
    # Add value labels on top of bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(counts)*0.01,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Customize the plot
    plt.title('Vehicle Type Distribution in IDMT Traffic Dataset', fontsize=16, fontweight='bold')
    plt.xlabel('Vehicle Type', fontsize=14)
    plt.ylabel('Number of Audio Files', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    
    # Add percentage labels
    total_files = sum(counts)
    for i, (label, count) in enumerate(zip(labels, counts)):
        percentage = (count / total_files) * 100
        plt.text(i, count/2, f'{percentage:.1f}%', ha='center', va='center', 
                fontweight='bold', color='white', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Histogram saved to: {save_path}")

def build_index(audio_dir, index_csv_path):
    """
    Build an index of all audio files and their labels.
    """
    data = []
    vehicle_counter = Counter()
    total_files = 0
    valid_files = 0
    channel_info = Counter()  # Track left/right channel distribution
    
    print(f"Building index from directory: {audio_dir}")
    
    for fname in os.listdir(audio_dir):
        if not fname.endswith('.wav'):
            continue
        
        total_files += 1
        vehicle, bg = parse_filename(fname)
        
        if vehicle:  # vehicle audio
            label = vehicle
            vehicle_counter[label] += 1
            valid_files += 1
            print(f"Added vehicle file: {label}")
            
            # Track original channel information before merging for statistics only
            original_vehicle_part = fname.split('_')[6] if len(fname.split('_')) > 6 else ''
            if original_vehicle_part.endswith('l'):
                channel_info['left'] += 1
            elif original_vehicle_part.endswith('r'):
                channel_info['right'] += 1
            else:
                channel_info['mono'] += 1
                
        elif bg:     # background
            label = bg
            vehicle_counter[label] += 1
            valid_files += 1
            print(f"Added background file: {label}")
        else:
            print(f"Skipped invalid file: {fname}")
            continue
            
        file_path = os.path.join(audio_dir, fname)
        data.append({'path': file_path, 'label': label})
        print("=" * 50)

    df = pd.DataFrame(data)
    df.to_csv(index_csv_path, index=False)
    
    # Print statistics
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Total files processed: {total_files}")
    print(f"Valid files added to dataset: {valid_files}")
    print(f"Invalid/skipped files: {total_files - valid_files}")
    print(f"Index saved to: {index_csv_path}")
    
    print("\nORIGINAL CHANNEL DISTRIBUTION (Before Merging):")
    print("-" * 50)
    for channel, count in channel_info.items():
        percentage = (count / valid_files) * 100 if valid_files > 0 else 0
        print(f"{channel.capitalize()}: {count} files ({percentage:.1f}%)")
    
    print("\nVEHICLE TYPE DISTRIBUTION (Left/Right Channels Merged):")
    print("-" * 55)
    for label, count in sorted(vehicle_counter.items()):
        percentage = (count / valid_files) * 100 if valid_files > 0 else 0
        print(f"{label}: {count} files ({percentage:.1f}%)")
    
    print(f"\nTotal unique vehicle types (after merging channels): {len(vehicle_counter)}")
    print("="*60)
    
    # Generate and save histogram with merged data
    plot_distribution_histogram(vehicle_counter)


class TrafficAudioDataset(Dataset):
    def __init__(self, csv_index_path, transform=None):
        """
        Args:
            csv_index_path (str): Path to the index csv file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = pd.read_csv(csv_index_path)
        self.transform = transform

        # Define a mapping from label to integer, for example:
        self.label_map = {'B': 0, 'C': 1, 'M': 2, 'T': 3, 'BG': 4}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path = self.data.iloc[idx]['path']
        label = self.data.iloc[idx]['label']
        label_idx = self.label_map[label]

        # Only load audio here, do not do any heavy preprocessing
        waveform, sample_rate = torchaudio.load(audio_path)
        if self.transform:
            waveform = self.transform(waveform)
        return waveform, label_idx

if __name__ == "__main__":
    # local path: /Users/xiangyifei/Documents/HPC_Efficient_Computing_System/dataset/IDMT_Traffic/audio
    build_index('/Users/xiangyifei/Documents/HPC_Efficient_Computing_System/dataset/IDMT_Traffic/audio', 'audio_index.csv')
