# LEAD Repository Setup

Follow these steps to set up and run the LEAD repository with the CAUEEG dataset loader:

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/DL4mHealth/LEAD.git
   ```

2. Move `caueeg_loader.py` into the dataset loader folder:
   ```sh
   mv caueeg_loader.py LEAD/data_provider/dataset_loader/
   ```
   *(This step may not be necessary, but ensure the file is in the correct location if needed.)*

3. Install the datasets from: https://drive.google.com/drive/folders/1KffFxezXzgIw-hseMmgLFy8xE31pXpYM 

## Modifications

3. Edit `LEAD/data_provider/data_loader.py`:
   - Inside the `data_folder_dict` dictionary, add the following entry:
     ```python
     'CAUEEG': CAUEEGLoader, # CAUEEG with 19 channels
     ```
   - Import the CAUEEGLoader by adding this line at the top:
     ```python
     from data_provider.dataset_loader.caueeg_loader import CAUEEGLoader
     ```

## Running the Code

Once the modifications are made, you should be able to use the CAUEEG data loader within the LEAD framework.

---
This README provides a minimal setup guide for enabling CAUEEG support in LEAD. Modify as needed based on your environment.

