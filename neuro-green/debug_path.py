#!/usr/bin/env python
"""
Debug CAUEEG data paths to find the mismatch between participants.tsv and folder names
"""

import os
import pandas as pd
import glob

def debug_paths(data_dir, derivatives_subdir):
    """Debug path issues in CAUEEG dataset"""
    
    # Set paths
    bids_root = data_dir
    # Remove leading slash for proper path joining
    derivatives_subdir_clean = derivatives_subdir.lstrip('/')
    derivatives_path = os.path.join(bids_root, derivatives_subdir_clean)
    participants_path = os.path.join(bids_root, 'participants.tsv')
    
    print(f"BIDS root: {bids_root}")
    print(f"Derivatives subdir (cleaned): {derivatives_subdir_clean}")
    print(f"Derivatives path: {derivatives_path}")
    print(f"Participants file: {participants_path}")
    print("-" * 80)
    
    # Check if paths exist
    print(f"BIDS root exists: {os.path.exists(bids_root)}")
    print(f"Derivatives path exists: {os.path.exists(derivatives_path)}")
    print(f"Participants file exists: {os.path.exists(participants_path)}")
    print("-" * 80)
    
    # Load participants data
    if os.path.exists(participants_path):
        participants_df = pd.read_csv(participants_path, sep='\t')
        print(f"Participants.tsv columns: {participants_df.columns.tolist()}")
        print(f"Number of participants: {len(participants_df)}")
        
        # Show first few participant IDs
        print("\nFirst 5 participant IDs in participants.tsv:")
        for i, pid in enumerate(participants_df['participant_id'].head()):
            print(f"  {i+1}: {pid}")
    else:
        print("ERROR: participants.tsv not found!")
        return
    
    print("-" * 80)
    
    # List actual folders in derivatives
    if os.path.exists(derivatives_path):
        folders = sorted([f for f in os.listdir(derivatives_path) if f.startswith('sub-')])
        print(f"\nActual folders in derivatives directory ({len(folders)} found):")
        for i, folder in enumerate(folders[:5]):
            print(f"  {i+1}: {folder}")
            
            # Check for EEG file (try both patterns)
            eeg_file_patterns = [
                os.path.join(derivatives_path, folder, 'eeg', 
                           f'{folder}_task-eyesClosed_desc-reject[]_eeg.fif'),
                os.path.join(derivatives_path, folder, 'eeg', 
                           f'{folder}_task-eyesClosed_desc-reject_eeg.fif')
            ]
            
            file_found = False
            for pattern in eeg_file_patterns:
                if os.path.exists(pattern):
                    print(f"     ✓ EEG file exists: {os.path.basename(pattern)}")
                    file_found = True
                    break
            
            if not file_found:
                print(f"     ✗ EEG file NOT found")
                print(f"       Tried patterns:")
                for pattern in eeg_file_patterns:
                    print(f"         - {pattern}")
    else:
        print("ERROR: derivatives directory not found!")
        return
    
    print("-" * 80)
    
    # Compare participant IDs with folder names
    print("\nChecking matches between participants.tsv and folders:")
    
    participants_ids = set(participants_df['participant_id'].tolist())
    folder_ids = set(folders)
    
    # Find matches
    matches = participants_ids.intersection(folder_ids)
    print(f"\nMatches found: {len(matches)}")
    
    # Find mismatches
    in_tsv_not_folders = participants_ids - folder_ids
    in_folders_not_tsv = folder_ids - participants_ids
    
    if in_tsv_not_folders:
        print(f"\nIn participants.tsv but NO folder ({len(in_tsv_not_folders)} items):")
        for pid in list(in_tsv_not_folders)[:5]:
            print(f"  - {pid}")
    
    if in_folders_not_tsv:
        print(f"\nFolders exist but NOT in participants.tsv ({len(in_folders_not_tsv)} items):")
        for folder in list(in_folders_not_tsv)[:5]:
            print(f"  - {folder}")
    
    # Suggest mapping
    print("\n" + "=" * 80)
    print("POSSIBLE SOLUTION:")
    print("=" * 80)
    
    # Check if it's a formatting issue
    if in_tsv_not_folders and in_folders_not_tsv:
        # Extract numbers from both sets
        tsv_numbers = []
        for pid in in_tsv_not_folders:
            try:
                num = int(pid.replace('sub-', ''))
                tsv_numbers.append(num)
            except:
                pass
        
        folder_numbers = []
        for folder in in_folders_not_tsv:
            try:
                num = int(folder.replace('sub-', ''))
                folder_numbers.append(num)
            except:
                pass
        
        if tsv_numbers and folder_numbers:
            print("It looks like there's a formatting mismatch!")
            print(f"Participants.tsv uses format like: {list(in_tsv_not_folders)[:3]}")
            print(f"Folders use format like: {list(in_folders_not_tsv)[:3]}")
            print("\nThe numbers might be the same but with different zero-padding.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python debug_caueeg_paths.py <data_dir> <derivatives_subdir>")
        print("Example: python debug_caueeg_paths.py /path/to/caueeg_bids /derivatives/sovaharmony")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    derivatives_subdir = sys.argv[2]
    
    debug_paths(data_dir, derivatives_subdir)