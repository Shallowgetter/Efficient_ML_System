"""
Analyze the data structure in the ADC dataset.
"""

FOLDER_PATH = "/Users/xiangyifei/Documents/HPC_Efficient_Computing_System/dataset/ADC"

import os
import numpy as np
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────
# Randomly choose one sample from FOLDER_PATH
# ──────────────────────────────────────────────
adcNpy = np.random.choice(os.listdir(FOLDER_PATH), 1)[0]

# Load the selected .npy file
adcData = np.load(os.path.join(FOLDER_PATH, adcNpy))
print("Loaded data from:", adcNpy)
# Print the shape of the loaded data
print(f"Shape of the loaded data: {adcData.shape}")
# Print the first few entries of the loaded data
print("First few entries of the loaded data:")
print(adcData[:5])
# Print the data type of the loaded data
print(f"Data type of the loaded data: {adcData.dtype}")
# Print the keys of the loaded data if it's a structured array
if isinstance(adcData, np.ndarray) and adcData.dtype.names:
    print("Keys in the structured array:")
    print(adcData.dtype.names)
else:
    print("The loaded data is not a structured array or does not have named fields.")

# Detailed analysis of complex data
print("\n" + "="*50)
print("Detailed Complex Data Analysis")
print("="*50)

# Extract real and imaginary parts
real_part = np.real(adcData)
imag_part = np.imag(adcData)

print(f"Real part range: [{np.min(real_part):.2f}, {np.max(real_part):.2f}]")
print(f"Imaginary part range: [{np.min(imag_part):.2f}, {np.max(imag_part):.2f}]")

# Calculate magnitude and phase
magnitude = np.abs(adcData)
phase = np.angle(adcData)

print(f"Magnitude range: [{np.min(magnitude):.2f}, {np.max(magnitude):.2f}]")
print(f"Phase range: [{np.min(phase):.2f}, {np.max(phase):.2f}] radians")

# Display detailed information of a sample
sample_idx = (0, 0, 0, 0)  # First sample
sample_value = adcData[sample_idx]
print(f"\nSample {sample_idx} value:")
print(f"  Complex form: {sample_value}")
print(f"  Real part: {np.real(sample_value):.2f}")
print(f"  Imaginary part: {np.imag(sample_value):.2f}")
print(f"  Magnitude: {np.abs(sample_value):.2f}")
print(f"  Phase: {np.angle(sample_value):.2f} radians")

# Visualize complex data
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('ADC Complex Data Analysis')

# Real part distribution
axes[0,0].hist(real_part.flatten(), bins=50, alpha=0.7)
axes[0,0].set_title('Real Part Distribution')
axes[0,0].set_xlabel('Real Part Value')
axes[0,0].set_ylabel('Frequency')

# Imaginary part distribution
axes[0,1].hist(imag_part.flatten(), bins=50, alpha=0.7, color='orange')
axes[0,1].set_title('Imaginary Part Distribution')
axes[0,1].set_xlabel('Imaginary Part Value')
axes[0,1].set_ylabel('Frequency')

# Magnitude distribution
axes[1,0].hist(magnitude.flatten(), bins=50, alpha=0.7, color='green')
axes[1,0].set_title('Magnitude Distribution')
axes[1,0].set_xlabel('Magnitude Value')
axes[1,0].set_ylabel('Frequency')

# Complex plane scatter plot (subset of data to avoid overcrowding)
sample_data = adcData[0, :10, 0, 0]  # Take a small subset of data
axes[1,1].scatter(np.real(sample_data), np.imag(sample_data), alpha=0.6)
axes[1,1].set_title('Complex Plane Representation (Sample Data)')
axes[1,1].set_xlabel('Real Part')
axes[1,1].set_ylabel('Imaginary Part')
axes[1,1].grid(True)

plt.tight_layout()
plt.show()