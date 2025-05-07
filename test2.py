import numpy as np

# Load the .npy file
data = np.load('/home/stud/timlin/bhome/DiffusionEEG/dataset/LDM_CAUEEG2/Label/old_label.npy')
# data = np.load('/home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG2/Label/label.npy')

# Basic information
print("Shape:", data.shape)
print("Data type:", data.dtype)
print("Size:", data.size)

# View some of the data
print("First few elements:", data[:10])  # Show first 10 elements

# Sort by the second column (ID, index 1)
sorted_data = data[data[:, 1].argsort()]

# Save the sorted data to a new file
np.save('/home/stud/timlin/bhome/DiffusionEEG/dataset/LDM_CAUEEG2/Label/label.npy', sorted_data)

print("Sorted data has been saved to 'label.npy'")

# Optional: Display first few rows of the sorted data
print("First few rows of sorted data:")
print(sorted_data[:10])


