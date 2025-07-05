"""window.py
Used for segmenting time series data into windows through overlapping segments.

Attention: Overlapping once used can only be applied to the last step before sent into the model.

Labels are mapped to 1 label per window within the window size using dominant label in the window.
"""

import numpy as np



def windows(data: np.ndarray, size: int, overlap: float):
    """
    Generate start and end indices for overlapping windows in the data.
    
    Args:
        data (np.ndarray): The input data array.
        size (int): The size of each window.
        overlap (float): The fraction of overlap between consecutive windows.
        
    Yields:
        tuple: Start and end indices for each window.
    """
    start = 0
    while start < len(data):
        yield int(start), int(start + size)
        start += int(size * (1 - overlap))  # Step size based on overlap


def segment_signal(data: np.ndarray, window_size: int, overlap: float):
    """
    Segment the input data into overlapping windows.
    
    Args:
        data (np.ndarray): The input data array with shape (N, F).
        window_size (int): The size of each window.
        overlap (float): The fraction of overlap between consecutive windows.
        
    Returns:
        np.ndarray: Segmented data with shape (M, window_size, F), where M is the number of segments.
    """
    segments = np.empty((0, window_size, data.shape[1]))
    count = 0
    for start, end in windows(data[:, 0], window_size, overlap):
        count += 1
        print(f'Segmentation: {count}')
        if end - start == window_size:
            segment = data[start:end]
            segments = np.vstack([segments, segment[np.newaxis, ...]])
    return segments

def segment_labels(data: np.ndarray, window_size: int, overlap: float):
    """
    Segment the labels corresponding to the input data into overlapping windows.
    
    Args:
        data (np.ndarray): The input data array with shape (N, F), where F includes the label in the last column.
        window_size (int): The size of each window.
        overlap (float): The fraction of overlap between consecutive windows.
        
    Returns:
        np.ndarray: Segmented labels with shape (M,), where M is the number of segments.
    """
    labels = np.empty((0,))
    for start, end in windows(data[:, 0], window_size, overlap):
        if end - start == window_size:
            label = np.mode(data[start:end, -1])[0][0]  # Assuming label is in the last column
            labels = np.append(labels, label)
    return labels