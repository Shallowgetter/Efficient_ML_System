"""
This file is to analysis the data sreucture of every type of data in eTraM dataset.
the whole local eTraM dataset folder sttructure is like this:
|-- eTraM/
        |-- Ego/ 
        |-- Static/
                |-- Eight_class_annotations/
                        |-- eight_class_annotations_test/
                        |-- eight_class_annotations_train/
                        |-- eight_class_annotations_val/
                |-- HDF5/
                        |-- test_h5_1/
				|-- test_day_001_bbox.npy
                               	|-- test_day_001_td.h5
                               	|-- test_day_002_bbox.npy
                               	|-- test_day_002_td.h5
				|-- …
                        |-- train_h5_1/
				|-- train_day_001_bbox.npy
                               	|-- train_day_001_td.h5
                               	|-- train_day_002_bbox.npy
                               	|-- train_day_002_td.h5
				|-- …
                        |-- val_h5_1/
				|-- val_day_001_bbox.npy
                               	|-- val_day_001_td.h5
                               	|-- val_day_002_bbox.npy
                               	|-- val_day_002_td.h5
				|-- …
                |-- RAW/
                        |-- test_1/
				|-- test_day_001_bbox.npy
                               	|-- test_day_001_td.raw
                               	|-- test_day_002_bbox.npy
                               	|-- test_day_002_td.raw
				|-- …
                        |-- train1/
				|-- train_day_001_bbox.npy
                               	|-- train_day_001_td.raw
                               	|-- train_day_002_bbox.npy
                               	|-- train_day_002_td.raw
				|-- …
                        |-- val_1/
				|-- val_day_001_bbox.npy
                               	|-- val_day_001_td.raw
                               	|-- val_day_002_bbox.npy
                               	|-- val_day_002_td.raw
				|-- …

Here we present different tools for different types of data in eTraM dataset.
"""

import numpy as np
import h5py

import h5py
import numpy as np
import os

def load_data(path: str, data_type: str) -> np.ndarray | dict:
    """
    This function loads data from eTraM dataset.
    Return a numpy array according to original data type (npy, h5, raw while raw currently not supported).
    For .h5 files with multiple datasets, it returns a dictionary where keys are dataset paths and values are numpy arrays.
    """
    if data_type == 'npy':
        return np.load(path)
    elif data_type == 'h5':
        # first check the file extension
        if not path.endswith('.h5'):
            raise ValueError("The file must be a .h5 file.")

        # Check if the file exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"The file '{path}' does not exist.")

        # check data structure
        print("Checking data structure of h5 file...")
        extracted_data = {} # Dictionary to store all extracted datasets

        try:
            with h5py.File(path, 'r') as f:
                keys = list(f.keys())
                print(f"Found top-level keys: {keys}")

                # Define a recursive function to explore groups and extract datasets
                def explore_and_extract(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        # If it's a dataset, extract its content to a NumPy array
                        print(f"  Extracting dataset: /{name} (Shape: {obj.shape}, Dtype: {obj.dtype})")
                        try:
                            extracted_data[name] = obj[()]
                        except Exception as e:
                            print(f"    Warning: Could not extract dataset /{name}: {e}")
                    elif isinstance(obj, h5py.Group):
                        # If it's a group, print it and continue exploring
                        print(f"  Exploring group: /{name}/")
                        # You can also access group attributes here if needed: obj.attrs
                
                # Use visititems to traverse all objects in the HDF5 file
                f.visititems(explore_and_extract)

        except Exception as e:
            raise IOError(f"Error reading HDF5 file '{path}': {e}")

        # Determine what to return based on the number of extracted datasets
        if not extracted_data:
            print(f"No datasets found in '{path}'. Returning an empty dictionary.")
            return {}
        elif len(extracted_data) == 1:
            # If there's only one dataset, return it directly as a NumPy array
            dataset_name = list(extracted_data.keys())[0]
            print(f"Only one dataset found ('{dataset_name}'). Returning it directly as a NumPy array.")
            return extracted_data[dataset_name]
        else:
            # If there are multiple datasets, return the dictionary
            print(f"Multiple datasets found. Returning a dictionary of NumPy arrays.")
            return extracted_data

    elif data_type == 'raw':
        raise NotImplementedError("Raw data type is not supported yet. Please convert it to npy or h5 format.")


def analysis_annotations(file: np.ndarray) -> None:
    """
    Analyze the annotations in the eTraM dataset.
    :param file: The annotation file to analyze.
    :return: A dictionary containing the analysis results.
    """
    if not isinstance(file, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    
    # Count unique labels
    unique_labels, counts = np.unique(file[:, 1], return_counts=True)
    print("Unique labels and their counts:")
    for label, count in zip(unique_labels, counts):
        print(f"Label {label}: {count} occurrences")

    
