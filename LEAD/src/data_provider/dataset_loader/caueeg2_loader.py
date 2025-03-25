import os
import numpy as np
import torch
from torch.utils.data import Dataset
import warnings
import random

warnings.filterwarnings('ignore')

def get_id_list_caueeg2(args, label_path, a=0.6, b=0.8):
    data_list = np.load(label_path)
    hc_list = list(data_list[np.where(data_list[:, 0] == 0)][:, 1])  # Healthy IDs
    mci_list = list(data_list[np.where(data_list[:, 0] == 1)][:, 1])  # MCI IDs
    dementia_list = list(data_list[np.where(data_list[:, 0] == 2)][:, 1])  # Dementia IDs
    
    if args.cross_val == 'fixed':
        random.seed(42)
    elif args.cross_val == 'mccv':
        random.seed(args.seed)
    elif args.cross_val == 'loso':
        all_ids = list(data_list[:, 1])
        hc_mci_dementia_list = sorted(hc_list + mci_list + dementia_list)
        test_ids = [hc_mci_dementia_list[(args.seed - 41) % len(hc_mci_dementia_list)]]
        train_ids = [id for id in hc_mci_dementia_list if id not in test_ids]
        random.seed(args.seed)
        random.shuffle(train_ids)
        val_ids = train_ids[int(0.9 * len(train_ids)):]
        return sorted(all_ids), sorted(train_ids), sorted(val_ids), sorted(test_ids)
    else:
        raise ValueError('Invalid cross_val. Please use fixed, mccv, or loso.')
    
    random.shuffle(hc_list)
    random.shuffle(mci_list)
    random.shuffle(dementia_list)
    
    all_ids = list(data_list[:, 1])
    train_ids = (hc_list[:int(a * len(hc_list))] +
                 mci_list[:int(a * len(mci_list))] +
                 dementia_list[:int(a * len(dementia_list))])
    val_ids = (hc_list[int(a * len(hc_list)):int(b * len(hc_list))] +
               mci_list[int(a * len(mci_list)):int(b * len(mci_list))] +
               dementia_list[int(a * len(dementia_list)):int(b * len(dementia_list))])
    test_ids = (hc_list[int(b * len(hc_list)):] +
                mci_list[int(b * len(mci_list)):] +
                dementia_list[int(b * len(dementia_list)):])
    
    return sorted(all_ids), sorted(train_ids), sorted(val_ids), sorted(test_ids)

def load_caueeg2_data_by_ids(data_path, label_path, ids):
    """Load CAUEEG2 data for specific subject IDs"""
    print(f"Loading data for subject IDs: {ids}")
    data_list = np.load(label_path)
    X, y = [], []
    
    for id in ids:
        # Find the label for this subject
        subject_indices = np.where(data_list[:, 1] == id)[0]
        if len(subject_indices) == 0:
            print(f"Subject ID {id} not found in label file")
            continue
            
        # Use the first matching label entry (should be only one)
        label = data_list[subject_indices[0], 0]
        
        # Try multiple file name patterns
        possible_filenames = [
            os.path.join(data_path, f"feature_{id:02d}.npy"),  # Main format from your script
            os.path.join(data_path, f"{id}.npy"),              # Alternative format
            os.path.join(data_path, f"feature_{id}.npy")       # No zero padding
        ]
        
        data_file = None
        for filename in possible_filenames:
            if os.path.exists(filename):
                data_file = filename
                break
                
        if data_file is None:
            print(f"No data file found for subject ID {id}")
            continue
            
        print(f"Loading data from {data_file}")
        data = np.load(data_file)
        print(f"Loaded data shape: {data.shape}")
        
        # If data has shape (epochs, times, channels)
        if len(data.shape) == 3:
            for epoch_idx in range(data.shape[0]):
                X.append(data[epoch_idx])
                # Store as [class, subject_id]
                y.append([label, id])
        else:
            print(f"Unexpected data shape for subject {id}: {data.shape}")
            continue
    
    # Handle empty dataset
    if len(X) == 0:
        print("Warning: No data found for any of the requested subject IDs")
        return np.array([]), np.array([]).reshape(0, 2)  # Empty arrays
    
    # Stack data and convert labels to array
    X = np.stack(X)
    y = np.array(y)
    
    print(f"Final dataset: X shape {X.shape}, y shape {y.shape}")
    return X, y

class CAUEEG2Loader(Dataset):
    def __init__(self, args, root_path, flag=None):
        print(f"Initializing CAUEEG2Loader with flag: {flag}")
        self.no_normalize = args.no_normalize
        self.root_path = root_path
        self.data_path = os.path.join(root_path, 'Feature')
        self.label_path = os.path.join(root_path, 'Label', 'label.npy')
        
        print(f"Data path: {self.data_path}")
        print(f"Label path: {self.label_path}")
        
        # Set default values for empty dataset case
        self.X = np.array([])
        self.y = np.array([]).reshape(0, 2)
        self.max_seq_len = 1000  # Default to expected time points
        
        # Check if label file exists
        if not os.path.exists(self.label_path):
            print(f"Warning: Label file not found at {self.label_path}")
            return
            
        a, b = 0.6, 0.8
        self.all_ids, self.train_ids, self.val_ids, self.test_ids = get_id_list_caueeg2(args, self.label_path, a, b)
        
        if flag == 'TRAIN':
            ids = self.train_ids
            print('train ids:', ids)
        elif flag == 'VAL':
            ids = self.val_ids
            print('val ids:', ids)
        elif flag == 'TEST':
            ids = self.test_ids
            print('test ids:', ids)
        elif flag == 'PRETRAIN':
            ids = self.all_ids
            print('all ids:', ids)
        else:
            raise ValueError('Invalid flag. Please use TRAIN, VAL, TEST, or PRETRAIN.')
        
        # Load data
        self.X, self.y = load_caueeg2_data_by_ids(self.data_path, self.label_path, ids)
        print(f"CAUEEG2 data shape: {self.X.shape}, {self.y.shape}")
        
        # Handle empty dataset
        if len(self.X) == 0:
            print(f"Warning: Empty dataset for {flag}")
            return
            
        # Merge Dementia into MCI class if needed
        self.y[:, 0] = np.where(self.y[:, 0] == 2, 1, self.y[:, 0])
        
        # Apply bandpass filtering if needed
        if hasattr(args, 'low_cut') and hasattr(args, 'high_cut') and hasattr(args, 'sampling_rate'):
            try:
                from data_provider.uea import bandpass_filter_func
                print(f"Applying bandpass filter: {args.low_cut}-{args.high_cut} Hz")
                
                filtered_X = []
                for i in range(self.X.shape[0]):
                    # Extract sample data (times, channels)
                    sample_data = self.X[i]
                    # Apply filter
                    filtered_sample = bandpass_filter_func(
                        sample_data, 
                        fs=args.sampling_rate, 
                        lowcut=args.low_cut, 
                        highcut=args.high_cut
                    )
                    filtered_X.append(filtered_sample)
                
                self.X = np.stack(filtered_X)
                print(f"After filtering: X shape {self.X.shape}")
            except Exception as e:
                print(f"Warning: Failed to apply bandpass filter: {e}")
        
        # Apply normalization if needed
        if not self.no_normalize:
            try:
                from data_provider.uea import normalize_batch_ts
                print("Applying normalization")
                
                normalized_X = []
                for i in range(self.X.shape[0]):
                    sample_data = self.X[i]
                    norm_sample = normalize_batch_ts(sample_data)
                    normalized_X.append(norm_sample)
                
                self.X = np.stack(normalized_X)
                print(f"After normalization: X shape {self.X.shape}")
            except Exception as e:
                print(f"Warning: Failed to apply normalization: {e}")
        
        # Set sequence length
        if len(self.X.shape) > 1:
            self.max_seq_len = self.X.shape[1]
        
        print(f"Loader initialized. Final X shape: {self.X.shape}, max_seq_len: {self.max_seq_len}")

    def __getitem__(self, index):
        if len(self.X) == 0:
            # Handle empty dataset case
            return torch.zeros(1, self.max_seq_len, 19).float(), torch.zeros(2).long()
            
        # Return one sample with shape (times, channels) and its label
        return torch.from_numpy(self.X[index]).float(), torch.from_numpy(self.y[index]).long()

    def __len__(self):
        return max(1, len(self.X))  # Ensure at least 1 to avoid DataLoader errors