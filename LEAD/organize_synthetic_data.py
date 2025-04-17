import os
import argparse
import numpy as np
import glob
import shutil
import re

def setup_folders(output_root):
    """Create the necessary folder structure"""
    # Create main folders
    feature_path = os.path.join(output_root, 'Feature')
    label_path = os.path.join(output_root, 'Label')
    
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(feature_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)
    
    return feature_path, label_path

def extract_subject_id(filename):
    """Extract subject ID from filename"""
    match = re.search(r'subject_(\d+)_eeg', filename)
    if match:
        return int(match.group(1))
    return None

def organize_multi_class_data(hc_dir, mci_dir, dementia_dir, output_root, start_id=1):
    """Organize synthetic data from multiple classes into CAUEEG format"""
    print(f"Organizing synthetic data into {output_root}")
    
    # Setup folders
    feature_path, label_path = setup_folders(output_root)
    
    # Dictionary to store subject mappings and labels
    subject_info = {}
    next_id = start_id
    
    # Process each class directory
    for class_dir, label in [(hc_dir, 0), (mci_dir, 1), (dementia_dir, 2)]:
        class_name = ["Healthy Control", "MCI", "Dementia"][label]
        print(f"\nProcessing {class_name} data from {class_dir}")
        
        # Find all npy files (excluding metadata)
        npy_files = [f for f in glob.glob(os.path.join(class_dir, "*.npy")) 
                    if not f.endswith("_metadata.npy")]
        
        print(f"Found {len(npy_files)} NPY files")
        
        # Copy files to Feature folder with appropriate naming
        for file_path in npy_files:
            filename = os.path.basename(file_path)
            original_subject_id = extract_subject_id(filename)
            
            if original_subject_id is not None:
                # Assign a new sequential ID
                new_subject_id = next_id
                next_id += 1
                
                # Store mapping and label
                subject_info[new_subject_id] = {
                    'original_id': original_subject_id,
                    'label': label,
                    'source': os.path.basename(class_dir),
                    'original_file': filename
                }
                
                # Define destination path with new ID
                dest_path = os.path.join(feature_path, f'feature_{new_subject_id:02d}.npy')
                
                # Copy the file
                shutil.copy2(file_path, dest_path)
                print(f"Copied {filename} (subject {original_subject_id}) to {dest_path} (new ID: {new_subject_id})")
    
    # Create labels
    # Each row is [label, subject_id]
    labels = np.array([[subject_info[sid]['label'], sid] for sid in sorted(subject_info.keys())], dtype='int32')
    
    # Save labels
    label_file = os.path.join(label_path, 'label.npy')
    np.save(label_file, labels)
    print(f"\nCreated label file with {len(labels)} subjects at {label_file}")
    
    # Save subject mapping for reference
    mapping_file = os.path.join(output_root, 'subject_mapping.txt')
    with open(mapping_file, 'w') as f:
        f.write("New ID\tOriginal ID\tClass\tSource\tOriginal File\n")
        for new_id, info in sorted(subject_info.items()):
            f.write(f"{new_id}\t{info['original_id']}\t{info['label']}\t{info['source']}\t{info['original_file']}\n")
    print(f"Created subject mapping file at {mapping_file}")
    
    # Summary
    class_counts = {}
    for info in subject_info.values():
        class_counts[info['label']] = class_counts.get(info['label'], 0) + 1
    
    print(f"\nOrganization complete:")
    print(f"- Total subjects: {len(subject_info)}")
    for label, count in sorted(class_counts.items()):
        class_name = ["Healthy Control", "MCI", "Dementia"][label]
        print(f"- {class_name} (class {label}): {count} subjects")
    print(f"\nOutput directory: {output_root}")

def main():
    parser = argparse.ArgumentParser(description="Organize multi-class synthetic EEG data into CAUEEG format")
    parser.add_argument("--hc_dir", type=str, required=True, 
                        help="Directory containing healthy control synthetic data")
    parser.add_argument("--mci_dir", type=str, required=True, 
                        help="Directory containing MCI synthetic data")
    parser.add_argument("--dementia_dir", type=str, required=True, 
                        help="Directory containing dementia synthetic data")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for organized dataset")
    parser.add_argument("--start_id", type=int, default=1,
                        help="Starting subject ID for the combined dataset")
    
    args = parser.parse_args()
    organize_multi_class_data(args.hc_dir, args.mci_dir, args.dementia_dir, 
                             args.output_dir, args.start_id)

if __name__ == "__main__":
    main()