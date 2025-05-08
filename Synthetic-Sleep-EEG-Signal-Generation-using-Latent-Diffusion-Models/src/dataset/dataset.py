import pandas as pd
import numpy as np
import torch
from monai.data import DataLoader, PersistentDataset
from monai.transforms import Compose, LoadImageD, ScaleIntensityD, EnsureChannelFirstD, Transform
from pandas import DataFrame

sfreq = 200
windows_size = 30 * sfreq

# Custom transform to format EEG data properly
class FormatEEGDataD(Transform):
    def __init__(self, keys):
        self.keys = keys
        
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in d:
                # Get the EEG data
                eeg_data = d[key]
                
                # Print original shape for debugging
                # print(f"Original shape: {eeg_data.shape}")
                
                # Check the shape and format accordingly
                if len(eeg_data.shape) == 4 and eeg_data.shape[0] == 1:
                    # If shape is [1, epochs, timepoints, channels]
                    if eeg_data.shape[3] == 19:
                        # Reshape to [epochs, timepoints, channels]
                        eeg_data = eeg_data.reshape(eeg_data.shape[1], eeg_data.shape[2], eeg_data.shape[3])
                        # Convert to [epochs, channels, timepoints]
                        eeg_data = eeg_data.permute(0, 2, 1)
                        # print(f"Reshaped from [1, epochs, timepoints, channels] to {eeg_data.shape}")
                    # If shape is [1, epochs, channels, timepoints]
                    elif eeg_data.shape[2] == 19:
                        # Reshape to [epochs, channels, timepoints]
                        eeg_data = eeg_data.reshape(eeg_data.shape[1], eeg_data.shape[2], eeg_data.shape[3])
                        # print(f"Reshaped from [1, epochs, channels, timepoints] to {eeg_data.shape}")
                elif len(eeg_data.shape) == 3:
                    # If shape is [epochs, timepoints, channels]
                    if eeg_data.shape[2] == 19:
                        # Convert to [epochs, channels, timepoints]
                        eeg_data = eeg_data.permute(0, 2, 1)
                        # print(f"Permuted from [epochs, timepoints, channels] to {eeg_data.shape}")
                    # If shape is already [epochs, channels, timepoints], keep as is
                    # elif eeg_data.shape[1] == 19:
                    #     print(f"Shape already correct: {eeg_data.shape}")
                
                # Ensure the data is in the format [epochs, channels, timepoints]
                assert len(eeg_data.shape) == 3 and eeg_data.shape[1] == 19, \
                    f"Expected shape [epochs, 19, timepoints], got {eeg_data.shape}"
                
                # Update the data
                d[key] = eeg_data
                
                # Print final shape for debugging
                # print(f"Final shape: {d[key].shape}")
        
        return d

def get_trans(dataset):
    if dataset == "caueeg2":
        transforms_list = Compose([
            # Load the data
            LoadImageD(keys='eeg', reader="NumpyReader", image_only=True),
            
            # Format the EEG data
            FormatEEGDataD(keys=['eeg']),
            
            # Normalize the data
            ScaleIntensityD(factor=1e6, keys='eeg'),  # For numeric stability
            ScaleIntensityD(minv=0, maxv=1, keys='eeg')  # Normalize to [0,1]
        ])
    else:
        # Original code for other datasets
        transforms_list = Compose([
            LoadImageD(keys='eeg'),
            EnsureChannelFirstD(keys='eeg'),
            ScaleIntensityD(factor=1e6, keys='eeg'),  # Numeric stability
            ScaleIntensityD(minv=0, maxv=1, keys='eeg'),  # Normalization
            RandSpatialCropD(keys='eeg', roi_size=[windows_size], random_size=False)
        ])
    return transforms_list


# Custom collate function to process one patient at a time
def patient_collate_fn(batch):
    # Since we are using batch_size=1, the batch will only have one element
    return batch[0]


def get_datalist(df: DataFrame, basepath: str, dataset: str):
    """
    Get data dicts for data loaders.
    """
    if dataset == "edfx":
        final = ".npy"
    else:
        final = ""

    data_dicts = []
    for index, row in df.iterrows():
        data_dicts.append({
            "eeg": f"{basepath}/{row['FILE_NAME_EEG']}{final}",
            "subject": float(row["subject"]),
            "night": float(row["night"]),
            "age": float(row["age"]),
            "gender": str(row["gender"]),
            "lightoff": str(row["LightsOff"]),
        })

    print(f"Found {len(data_dicts)} subjects.")
    return data_dicts


def get_caueeg2_datalist(base_path, label_filter=None):
    """
    Create data dictionaries for CAUEEG2 dataset
    """
    import glob
    import os
    import numpy as np
    
    # Find all feature files
    feature_files = glob.glob(os.path.join(base_path, 'Feature', 'feature_*.npy'))
    feature_files.sort()
    
    if not feature_files:
        raise ValueError(f"No feature files found in {os.path.join(base_path, 'Feature')}. Check the path.")
    
    # Load labels
    labels_path = os.path.join(base_path, 'Label', 'label.npy')
    if os.path.exists(labels_path):
        try:
            labels = np.load(labels_path)
            print(f"Loaded labels with shape: {labels.shape}")
            
            # Create a dictionary mapping subject_id to label
            label_dict = {int(subject_id): int(label) for label, subject_id in labels}
            print(f"Created label dictionary with {len(label_dict)} entries")
            
            # Process label filter if provided
            if label_filter is not None:
                if isinstance(label_filter, (list, tuple)):
                    # Convert string labels to integers if needed
                    numeric_filters = []
                    for filt in label_filter:
                        if filt == 'hc' or filt == 'healthy' or filt == 'healthy controls':
                            numeric_filters.append(0)
                        elif filt == 'mci':
                            numeric_filters.append(1)
                        elif filt == 'dementia':
                            numeric_filters.append(2)
                        elif isinstance(filt, (int, float)):
                            numeric_filters.append(int(filt))
                    label_filter = numeric_filters
                else:
                    # Single filter value
                    if label_filter == 'hc' or label_filter == 'healthy' or label_filter == 'healthy controls':
                        label_filter = [0]
                    elif label_filter == 'mci':
                        label_filter = [1]
                    elif label_filter == 'dementia':
                        label_filter = [2]
                    elif isinstance(label_filter, (int, float)):
                        label_filter = [int(label_filter)]
                    else:
                        try:
                            label_filter = [int(label_filter)]
                        except:
                            print(f"Warning: Unrecognized label filter '{label_filter}'. Using all labels.")
                            label_filter = None
        except Exception as e:
            print(f"Error loading labels: {e}")
            label_dict = {}
            label_filter = None
    else:
        print(f"Warning: Labels file not found at {labels_path}")
        label_dict = {}
        label_filter = None
    
    # Analyze available labels before filtering
    all_subject_labels = {}
    for file_path in feature_files:
        filename = os.path.basename(file_path)
        subject_id = int(filename.split('_')[1].split('.')[0])
        label = label_dict.get(subject_id, -1)
        all_subject_labels[subject_id] = label
    
    # Count available labels
    label_counts = {}
    for subject_id, label in all_subject_labels.items():
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1
            
    # Show available labels
    label_names = {0: 'Healthy', 1: 'MCI', 2: 'Dementia', -1: 'Unknown'}
    print("Available labels in dataset:")
    for label, count in label_counts.items():
        print(f"  - {label_names.get(label, f'Unknown-{label}')} (label={label}): {count} subjects")
    
    # Build filtered dataset
    data_dicts = []
    for file_path in feature_files:
        # Extract subject_id from filename (e.g., feature_01.npy -> 1)
        filename = os.path.basename(file_path)
        subject_id = int(filename.split('_')[1].split('.')[0])
        
        # Get label for this subject
        label = label_dict.get(subject_id, -1)  # -1 if not found
        
        # Skip if it doesn't match the filter
        if label_filter is not None and label not in label_filter:
            continue
        
        data_dict = {
            "eeg": file_path,
            "subject": subject_id,
            "label": label
        }
        data_dicts.append(data_dict)
    
    # Print summary of selected data
    if label_filter is not None:
        included_labels = [label_names.get(l, f"Unknown-{l}") for l in label_filter]
        
        # Check if we have an empty dataset after filtering
        if not data_dicts:
            print(f"ERROR: No data found matching label filter: {', '.join(included_labels)}")
            print("Please choose from the available labels listed above.")
            print("Example usage: --label_filter hc  or  --label_filter mci  or  --label_filter 0,1")
            
            # Instead of raising an error, return an empty list
            # The calling function can handle this case
            return []
            
        print(f"Selected {len(data_dicts)} files with labels: {', '.join(included_labels)}")
    else:
        print(f"Found {len(data_dicts)} files for CAUEEG2 dataset (all labels).")
    
    # Try to load one file to check its shape
    if data_dicts:
        try:
            sample_data = np.load(data_dicts[0]["eeg"])
            print(f"Sample data shape: {sample_data.shape}")
        except Exception as e:
            print(f"Could not load sample file: {e}")
    
    return data_dicts


def train_dataloader(config, args, transforms_list, dataset):
    if dataset == "caueeg2":
        # Use the function for CAUEEG2 with label filtering
        label_filter = args.label_filter if hasattr(args, 'label_filter') else None
        all_dicts = get_caueeg2_datalist(args.path_pre_processed, label_filter)
        
        # Handle empty dataset case
        if not all_dicts:
            raise ValueError("Training dataset is empty after applying label filter. Check available labels using the debug script.")
            
        # Use the first 80% for training
        train_size = int(0.8 * len(all_dicts))
        train_dicts = all_dicts[:train_size]
        print(f"Using {len(train_dicts)} files for training")
    else:
        # Original code for other datasets
        train_df = pd.read_csv(args.path_train_ids)
        train_dicts = get_datalist(train_df, basepath=args.path_pre_processed, dataset=dataset)

    train_ds = PersistentDataset(data=train_dicts,
                               transform=transforms_list,
                               cache_dir=None)

    # Use batch_size=1 and our custom collate function to process one patient at a time
    train_loader = DataLoader(
        train_ds,
        batch_size=1,  # Process one patient at a time
        shuffle=True,
        num_workers=config.train.num_workers,
        drop_last=config.train.drop_last,
        pin_memory=False,
        persistent_workers=True,
        collate_fn=patient_collate_fn  # Custom collate function
    )
    return train_loader


def valid_dataloader(config, args, transforms_list, dataset):
    if dataset == "caueeg2":
        # Use the function for CAUEEG2 with label filtering
        label_filter = args.label_filter if hasattr(args, 'label_filter') else None
        all_dicts = get_caueeg2_datalist(args.path_pre_processed, label_filter)
        
        # Handle empty dataset case
        if not all_dicts:
            raise ValueError("Validation dataset is empty after applying label filter. Check available labels using the debug script.")
            
        # Use the last 20% for validation
        train_size = int(0.8 * len(all_dicts))
        valid_dicts = all_dicts[train_size:]
        print(f"Using {len(valid_dicts)} files for validation")
    else:
        # Original code for other datasets
        valid_df = pd.read_csv(args.path_valid_ids)
        valid_dicts = get_datalist(valid_df, basepath=args.path_pre_processed, dataset=dataset)

    valid_ds = PersistentDataset(data=valid_dicts, transform=transforms_list,
                               cache_dir=None)

    # Use batch_size=1 and our custom collate function to process one patient at a time
    valid_loader = DataLoader(
        valid_ds, 
        batch_size=1,  # Process one patient at a time
        shuffle=True,
        num_workers=config.train.num_workers, 
        drop_last=config.train.drop_last,
        pin_memory=False,
        persistent_workers=True,
        collate_fn=patient_collate_fn  # Custom collate function
    )

    return valid_loader


def test_dataloader(config, args, transforms_list, dataset, upper_limit=None):
    test_df = pd.read_csv(args.path_test_ids)

    if upper_limit is not None:
        test_df = test_df[:upper_limit]

    test_dicts = get_datalist(test_df, basepath=args.path_pre_processed, dataset=dataset)

    test_ds = PersistentDataset(data=test_dicts, transform=transforms_list,
                                 cache_dir=None)

    # Use batch_size=1 and our custom collate function to process one patient at a time
    test_loader = DataLoader(
        test_ds, 
        batch_size=1,  # Process one patient at a time
        shuffle=True,
        num_workers=config.train.num_workers, 
        drop_last=config.train.drop_last,
        pin_memory=False,
        persistent_workers=True,
        collate_fn=patient_collate_fn  # Custom collate function
    )

    return test_loader