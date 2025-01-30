Google Disk: https://drive.google.com/drive/folders/1qvIyEF0INjIYH-uF4PS9KQdnCx9gnq0H
# EEG Harmonization Repository

A repository for EEG harmonization and preprocessing tasks, including EEGFormer implementations and data
visualization tools.

## Quick Start
This is needed for the environment used for preprocessing
```bash
git clone https://github.com/GRUNECO/eeg_harmonization.git
pip install -r eeg_harmonization/requirements.txt
pip install -e eeg_harmonization/
```

## Environment Setup

1. **Activate your conda environment**:
   ```bash
   conda activate your_env_name
   ```

2. **Modify preprocess_run.sh** to use your active conda environment.

## Key Components

- **EEGFormer Implementation**: Located in the `/EEGFormer` directory.
- **Data Visualization**:
  - `eeg_plot.png`: Raw preprocessed EEG (unfiltered)
  - `filtered_eeg_epochs.png`: Band-limited (60Hz) EEG epochs
- **Dataset Handling**: `EEGSyndromeDataset.py` connects participant epochs with diagnostic labels.
- **Modified EEGFormer Code**: Includes `models.py`, `train.py`, and `utils.py` adapted from the original
implementation.
- **Training**: Currently uses dummy data instead of CSV input.
- **Preprocessing**:
  - `preprocess_run.sh`: Script to run preprocessing
  - Preprocessing details: `preprocessing_feature_extraction_eeg.ipynb`
- **Participant Categorization**: Scripts in `categorize_participants/` sort participants into:
  - `dementia_subjects.txt`
  - `nc_smc_subjects.txt`
  - `mci_subjects.txt`
- **Plot scripts**:
  - `plot_eeg.py`: For visualizing EEG data


## Notes

- **Requirements**: See `requirements.yaml` for the full list.
- **Preprocessing Details**: Review `preprocessing_feature_extraction_eeg.ipynb` for specific steps.

## EEGFormer Modifications

- Active development in a dedicated folder, with updated code available soon.

## Plotting Tools

- **plot_eeg.py**: Visualizes raw and filtered EEG data