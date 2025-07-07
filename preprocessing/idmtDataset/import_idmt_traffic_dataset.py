""" Import script for IDMT-Traffic dataset
Ref:
    J. Abeßer, S. Gourishetti, A. Kátai, T. Clauß, P. Sharma, J. Liebetrau: IDMT-Traffic: An Open Benchmark
    Dataset for Acoustic Traffic Monitoring Research, EUSIPCO, 2021
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

__author__ = 'Jakob Abeßer (jakob.abesser@idmt.fraunhofer.de)'


def import_idmt_traffic_dataset(fn_txt: str = "idmt_traffic_all") -> pd.DataFrame:
    """ Import IDMT-Traffic dataset
    Args:
        fn_txt (str): Text file with all WAV files
    Returns:
        df_dataset (pd.Dataframe): File-wise metadata
            Columns:
                'file': WAV filename,
                'is_background': True if recording contains background noise (no vehicle), False else
                'date_time': Recording time (YYYY-MM-DD-HH-mm)
                'location': Recording location
                'speed_kmh': Speed limit at recording site (km/h), UNK if unknown,
                'sample_pos': Sample position (centered) within the original audio recording,
                'daytime': M(orning) or (A)fternoon,
                'weather': (D)ry or (W)et road condition,
                'vehicle': (B)us, (C)ar, (M)otorcycle, or (T)ruck,
                'source_direction': Source direction of passing vehicle: from (L)eft or from (R)ight,
                'microphone': (SE)= (high-quality) sE8 microphones, (ME) = (low-quality) MEMS microphones (ICS-43434),
                'channel': Original stereo pair channel (12) or (34)
    """
    # load file list
    df_files = pd.read_csv(fn_txt, names=('file',))
    fn_file_list = df_files['file'].to_list()

    # load metadata from file names
    df_dataset = []

    for f, fn in enumerate(fn_file_list):
        fn = fn.replace('.wav', '')
        parts = fn.split('_')

        # background noise files
        if '-BG' in fn:
            date_time, location, speed_kmh, sample_pos, mic, channel = parts
            vehicle, source_direction, weather, daytime = 'None', 'None', 'None', 'None'
            is_background = True

        # files with vehicle passings
        else:
            date_time, location, speed_kmh, sample_pos, daytime, weather, vehicle_direction, mic, channel = parts
            vehicle, source_direction = vehicle_direction
            is_background = False

        channel = channel.replace('-BG', '')
        speed_kmh = speed_kmh.replace('unknownKmh', 'UNK')
        speed_kmh = speed_kmh.replace('Kmh', '')

        df_dataset.append({'file': fn,
                           'is_background': is_background,
                           'date_time': date_time,
                           'location': location,
                           'speed_kmh': speed_kmh,
                           'sample_pos': sample_pos,
                           'daytime': daytime,
                           'weather': weather,
                           'vehicle': vehicle,
                           'source_direction': source_direction,
                           'microphone': mic,
                           'channel': channel})

    df_dataset = pd.DataFrame(df_dataset, columns=('file', 'is_background', 'date_time', 'location', 'speed_kmh', 'sample_pos', 'daytime', 'weather', 'vehicle',
                                                   'source_direction', 'microphone', 'channel'))

    return df_dataset


def analyze_vehicle_distribution(df_dataset, dataset_name):
    """
    Analyze and visualize vehicle type distribution
    
    Args:
        df_dataset (pd.DataFrame): Dataset to analyze
        dataset_name (str): Name of the dataset (for titles and prints)
    """
    # Filter out background files (no vehicles)
    vehicle_data = df_dataset[df_dataset['is_background'] == False]
    
    # Count vehicle types
    vehicle_counts = vehicle_data['vehicle'].value_counts()
    
    # Vehicle type mapping for better display
    vehicle_mapping = {
        'B': 'Bus',
        'C': 'Car', 
        'M': 'Motorcycle',
        'T': 'Truck'
    }
    
    print(f"\n{'='*60}")
    print(f"Vehicle Distribution Analysis - {dataset_name}")
    print(f"{'='*60}")
    print(f"Total samples with vehicles: {len(vehicle_data)}")
    print(f"Background samples: {len(df_dataset[df_dataset['is_background'] == True])}")
    print(f"Total samples: {len(df_dataset)}")
    print(f"\nVehicle Type Distribution:")
    print(f"{'-'*40}")
    
    # Print vehicle counts
    for vehicle_code in ['B', 'C', 'M', 'T']:
        count = vehicle_counts.get(vehicle_code, 0)
        percentage = (count / len(vehicle_data) * 100) if len(vehicle_data) > 0 else 0
        print(f"{vehicle_mapping[vehicle_code]:<12}: {count:>6} samples ({percentage:>5.1f}%)")
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    
    # Prepare data for plotting
    vehicle_types = [vehicle_mapping[code] for code in ['B', 'C', 'M', 'T']]
    counts = [vehicle_counts.get(code, 0) for code in ['B', 'C', 'M', 'T']]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    # Create bar plot
    bars = plt.bar(vehicle_types, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Customize the plot
    plt.title(f'Vehicle Type Distribution - {dataset_name}', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Vehicle Type', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Samples', fontsize=12, fontweight='bold')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add total count in the title area
    plt.text(0.5, 0.95, f'Total Vehicle Samples: {len(vehicle_data)}', 
             transform=plt.gca().transAxes, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return vehicle_counts


def compare_datasets(train_counts, test_counts):
    """
    Create a comparison plot between training and test datasets
    
    Args:
        train_counts (pd.Series): Training set vehicle counts
        test_counts (pd.Series): Test set vehicle counts
    """
    vehicle_mapping = {
        'B': 'Bus',
        'C': 'Car', 
        'M': 'Motorcycle',
        'T': 'Truck'
    }
    
    # Prepare data
    vehicle_types = [vehicle_mapping[code] for code in ['B', 'C', 'M', 'T']]
    train_data = [train_counts.get(code, 0) for code in ['B', 'C', 'M', 'T']]
    test_data = [test_counts.get(code, 0) for code in ['B', 'C', 'M', 'T']]
    
    # Create comparison plot
    x = np.arange(len(vehicle_types))
    width = 0.35
    
    plt.figure(figsize=(12, 7))
    
    bars1 = plt.bar(x - width/2, train_data, width, label='Training Set', 
                    color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = plt.bar(x + width/2, test_data, width, label='Test Set', 
                    color='#e74c3c', alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # Customize the plot
    plt.title('Vehicle Type Distribution Comparison: Training vs Test Set', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Vehicle Type', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Samples', fontsize=12, fontweight='bold')
    plt.xticks(x, vehicle_types)
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.show()
    
    # Print comparison statistics
    print(f"\n{'='*60}")
    print(f"Training vs Test Set Comparison")
    print(f"{'='*60}")
    print(f"{'Vehicle Type':<12} {'Training':<10} {'Test':<10} {'Ratio (Train/Test)':<15}")
    print(f"{'-'*50}")
    
    for code, vehicle_name in vehicle_mapping.items():
        train_count = train_counts.get(code, 0)
        test_count = test_counts.get(code, 0)
        ratio = train_count / test_count if test_count > 0 else float('inf')
        print(f"{vehicle_name:<12} {train_count:<10} {test_count:<10} {ratio:<15.2f}")


if __name__ == '__main__':
    import sys
    
    # Custom folder path input
    if len(sys.argv) > 1:
        # Get folder path from command line arguments
        folder_path = sys.argv[1]
    else:
        # Interactive folder path input
        folder_path = input("Please enter the folder path containing txt files: ").strip()
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder does not exist: {folder_path}")
        sys.exit(1)
    
    # txt file list
    fn_txt_list = ["idmt_traffic_all.txt",    # complete IDMT-Traffic dataset
                   "eusipco_2021_train.txt",  # training set of EUSIPCO 2021 paper
                   "eusipco_2021_test.txt"]   # test set of EUSIPCO 2021 paper

    # Build complete file paths and check if files exist
    existing_files = []
    for fn_txt in fn_txt_list:
        full_path = os.path.join(folder_path, fn_txt)
        if os.path.exists(full_path):
            existing_files.append(full_path)
        else:
            print(f"Warning: File does not exist: {full_path}")
    
    if not existing_files:
        print("Error: No target txt files found in the specified folder")
        sys.exit(1)
    
    print(f"Found {len(existing_files)} txt files in folder '{folder_path}'")
    
    # Store datasets for comparison
    datasets = {}
    vehicle_counts = {}
    
    # Import metadata and analyze vehicle distribution
    for full_path in existing_files:
        fn_txt = os.path.basename(full_path)
        print(f'\n{"="*60}')
        print(f'Processing file: {fn_txt}')
        print(f'Full path: {full_path}')
        print(f'{"="*60}')
        
        try:
            df_result = import_idmt_traffic_dataset(full_path)
            print(f'Successfully loaded {len(df_result)} records')
            
            # Store dataset
            dataset_name = fn_txt.replace('.txt', '')
            datasets[dataset_name] = df_result
            
            # Analyze vehicle distribution for training and test sets
            if 'train' in fn_txt or 'test' in fn_txt:
                vehicle_counts[dataset_name] = analyze_vehicle_distribution(df_result, dataset_name)
            
        except Exception as e:
            print(f'Failed to load file: {e}')
    
    # Create comparison plot if both training and test sets are available
    if 'eusipco_2021_train' in vehicle_counts and 'eusipco_2021_test' in vehicle_counts:
        compare_datasets(vehicle_counts['eusipco_2021_train'], 
                        vehicle_counts['eusipco_2021_test'])
