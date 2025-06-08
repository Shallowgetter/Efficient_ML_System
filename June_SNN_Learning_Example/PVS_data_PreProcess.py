'''
This py file aims to preprocess specific data from PVS dataset.

Needed files:
- dataset_mpu_left.csv
- dataset_mpu_right.csv
- dataset_gps_mpu_left.csv
- dataset_gps_mpu_right.csv

Needed columns:
- 'timestamp'
- 'mag_x_dashboard', 'mag_y_dashboard', 'mag_z_dashboard'
- 'mag_x_above_suspension', 'mag_y_above_suspension', 'mag_z_above_suspension'

Function of this file:
- Load the data from the specified CSV files.
- Extract the required columns.
- Save columns to numpy arrays.
- Save the numpy arrays to .npy files.
'''

import pandas as pd
import numpy as np

def preprocess_pvs_data(file_path: str, save_path: str):
    # Load data from CSV file
    data = pd.read_csv(file_path)
    print(f"Loaded data from {file_path} with shape: {data.shape}")
    # Extract required columns
    columns = ['timestamp', 'mag_x_dashboard', 'mag_y_dashboard', 'mag_z_dashboard',
               'mag_x_above_suspension', 'mag_y_above_suspension', 'mag_z_above_suspension']
    data = data[columns]
    print(f"Extracted columns: {columns}")
    # Convert to numpy array
    data_array = data.to_numpy()
    # Save to .npy file
    np.save(save_path, data_array)
    print(f"Saved data to {save_path} with shape: {data_array.shape}")

if __name__ == "__main__":
    # Define file paths
    left_file_path = r'data\PVS\dataset_mpu_left.csv'
    right_file_path = r'data\PVS\dataset_mpu_right.csv'
    gps_left_file_path = r'data\PVS\dataset_gps_mpu_left.csv'
    gps_right_file_path = r'data\PVS\dataset_gps_mpu_right.csv'

    # Define save paths
    left_save_path = r'data\PVS_npy\pvs_data_left.npy'
    right_save_path = r'data\PVS_npy\pvs_data_right.npy'
    gps_left_save_path = r'data\PVS_npy\pvs_gps_data_left.npy'
    gps_right_save_path = r'data\PVS_npy\pvs_gps_data_right.npy'

    # Preprocess and save data
    preprocess_pvs_data(left_file_path, left_save_path)
    preprocess_pvs_data(right_file_path, right_save_path)
    preprocess_pvs_data(gps_left_file_path, gps_left_save_path)
    preprocess_pvs_data(gps_right_file_path, gps_right_save_path)