import os
import numpy as np
import glob

# Paths
synthetic_path = "dataset/SYNTH-CAUEEG2/Feature"

def transpose_files(feature_dir):
    """
    Transpose synthetic data from (1000, 19, x) to (x, 1000, 19)
    to match the genuine data format
    """
    print(f"Searching for files in {feature_dir}")
    feature_files = glob.glob(os.path.join(feature_dir, "*.npy"))
    print(f"Found {len(feature_files)} files to process")
    
    # Create a backup directory
    backup_dir = os.path.join(os.path.dirname(feature_dir), "Feature_Backup")
    os.makedirs(backup_dir, exist_ok=True)
    
    success_count = 0
    error_count = 0
    
    for file_path in feature_files:
        filename = os.path.basename(file_path)
        try:
            # Load data
            data = np.load(file_path)
            original_shape = data.shape
            
            # Check if already in correct shape
            if len(data.shape) == 3 and data.shape[0] < data.shape[1] and data.shape[1] == 1000 and data.shape[2] == 19:
                print(f"File {filename} already has the correct shape {data.shape}. Skipping.")
                continue
                
            # Backup original file
            backup_path = os.path.join(backup_dir, filename)
            np.save(backup_path, data)
            
            # Verify that the data is in (1000, 19, x) format
            if len(data.shape) == 3 and data.shape[0] == 1000 and data.shape[1] == 19:
                # Transpose from (1000, 19, x) to (x, 1000, 19)
                transposed_data = np.transpose(data, (2, 0, 1))
                print(f"Transposing {filename}: {original_shape} → {transposed_data.shape}")
                
                # Save transposed data back to original location
                np.save(file_path, transposed_data)
                success_count += 1
            else:
                print(f"Warning: File {filename} has unexpected shape {data.shape}. Should be (1000, 19, x).")
                error_count += 1
                
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            error_count += 1
    
    print(f"\nTransposition complete:")
    print(f"  Successfully transposed: {success_count} files")
    print(f"  Errors/Skipped: {error_count} files")
    print(f"  Original files backed up to: {backup_dir}")

def check_shapes(feature_dir):
    """Check shapes of all files to verify they match the expected format"""
    print(f"\nVerifying file shapes in {feature_dir}")
    feature_files = glob.glob(os.path.join(feature_dir, "*.npy"))
    
    if not feature_files:
        print("No files found!")
        return
        
    shapes = {}
    for file_path in feature_files:
        try:
            data = np.load(file_path)
            shape = data.shape
            if shape in shapes:
                shapes[shape] += 1
            else:
                shapes[shape] = 1
        except Exception as e:
            print(f"Error checking {os.path.basename(file_path)}: {str(e)}")
    
    print("Shape distribution after transposition:")
    for shape, count in sorted(shapes.items(), key=lambda x: x[1], reverse=True):
        print(f"  Shape {shape}: {count} files ({count/len(feature_files)*100:.1f}%)")
    
    # Check if all shapes have the correct format (x, 1000, 19)
    correct_format = True
    for shape in shapes.keys():
        if len(shape) != 3 or shape[1] != 1000 or shape[2] != 19:
            correct_format = False
            print(f"  Warning: Shape {shape} doesn't match the expected format (x, 1000, 19)")
    
    if correct_format:
        print("All files have the correct shape format (x, 1000, 19)!")
    else:
        print("Some files don't have the correct shape format (x, 1000, 19).")

if __name__ == "__main__":
    # First transpose all files
    transpose_files(synthetic_path)
    
    # Then verify the shapes
    check_shapes(synthetic_path)
    
    print("\nProcess complete. The synthetic data should now match the shape of the genuine data.")