# Instructions to Compile and Run System

## Prerequisites

- Ensure you have the `data` folder available in your working directory (contact the supervisors to obtain a copy of the eyes closed data)
- Access to the UiS SLURM servers (for `sbatch` commands)
- Git installed on your system
- Conda/Anaconda environment manager

## Step 1: Preprocessing

The preprocessing step is essential for preparing the EEG data for subsequent analysis and model training. Follow these steps in order:

### 1.1 Clone EEG Harmonization Repository

Clone the required EEG harmonization dependency:

```bash
git clone https://github.com/GRUNECO/eeg_harmonization.git
```

### 1.2 Create Conda Environment

Create the required conda environment using the provided script:

```bash
sbatch preprocess/create_env.sh
```

**Note**: Wait for this job to complete successfully before proceeding to the next step. You can monitor the job status using:

```bash
squeue -u $USER
```

To monitor the job progress in real-time, you can also follow the output log:

```bash
tail -f outputs/env_setup.out
```

### 1.3 Configure User Path

Before running the preprocessing pipeline, you need to update the user path in the preprocessing script to match your system:

Update line 4 in `preprocess/preprocess.py`:
```python
user_path = "/home/stud/timlin/bhome/DiffusionEEG"
```
Change this to your actual DiffusionEEG base directory path.

### 1.4 Run Preprocessing Pipeline

Once the conda environment has been successfully created and the user path has been configured, execute the main preprocessing pipeline:

```bash
sbatch preprocess/run_preprocess.sh
```

This script will handle the complete preprocessing workflow for your EEG data.

### 1.5 Configure CAUEEG2 Paths

After the initial preprocessing completes, you need to configure the paths in the CAUEEG2 processing script:

Update the root paths in `preprocess/caueeg2.py` on lines 18 and 21:
```python
# Line 18:
root = '/home/stud/timlin/bhome/DiffusionEEG/data/caueeg_bids'

# Line 21:
output_root = '/home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG2'
```

Replace `/home/stud/timlin/bhome/DiffusionEEG` with your actual DiffusionEEG base directory path.

### 1.6 Run CAUEEG2 Processing

Execute the CAUEEG2 processing pipeline:

```bash
sbatch preprocess/run_caueeg2.sh
```

### 1.7 Verify Preprocessing Completion

After the preprocessing job completes, verify that the output files have been generated in the expected directories. Check the SLURM output logs for any errors or warnings.

---

**Expected Output**: Upon successful completion of the preprocessing step, you should have processed EEG data ready for the subsequent analysis and model training phases.

**Troubleshooting**: If you encounter issues during preprocessing, check the SLURM job logs in your output directory for detailed error messages.

## Step 2: LEAD-based Diffusion Model Training

### 2.1 Create LEAD Environment

First, create the required conda environment for LEAD:

```bash
sbatch LEAD/create_env.sh
```

### 2.2 Run LEAD Training (Multiple Health Filters)

The LEAD diffusion model needs to be trained separately for different health conditions. You will run the training script 3 times, each time with a different health filter.

Before each run, update line 20 in `LEAD/run_LEAD.sh`:
```bash
HEALTH_FILTER="dementia"  # Change this value for each run
```

The available health filter options are:
- `"hc"` - healthy controls only
- `"mci"` - MCI patients only  
- `"dementia"` - dementia patients only

**Run the training for each health filter:**

```bash
# First run (e.g., healthy controls)
# Update HEALTH_FILTER="hc" in LEAD/run_LEAD.sh line 20
sbatch LEAD/run_LEAD.sh

# Second run (e.g., MCI patients)  
# Update HEALTH_FILTER="mci" in LEAD/run_LEAD.sh line 20
sbatch LEAD/run_LEAD.sh

# Third run (e.g., dementia patients)
# Update HEALTH_FILTER="dementia" in LEAD/run_LEAD.sh line 20
sbatch LEAD/run_LEAD.sh
```

**Note**: Since the output files include job IDs (`%j`), you can run all three jobs simultaneously after updating the health filter for each submission. Each job will have a unique output file like `outputs/Simplified_Diffusion_ArcMargin_LR_0001_SS_1000_DS2_[JOB_ID].out`.

### 2.3 Monitor LEAD Training

Monitor the training progress using:
```bash
squeue -u $USER
```

To follow the training output in real-time:
```bash
tail -f outputs/Simplified_Diffusion_ArcMargin_LR_0001_SS_1000_DS2_[JOB_ID].out
```

### 2.4 Generate Synthetic Data

After training is complete, the model weights will be saved in `checkpoints/LEAD/diffusion/LEAD/`. You can now generate synthetic EEG data for each health condition.

For each trained model, update the following parameters in `generate_data.sh`:

1. **Checkpoint path** - Update to point to the specific trained model:
```bash
--checkpoints_path "checkpoints/LEAD/diffusion/LEAD/Simplified-Diffusion-LR-0001-SS-1000-DS2-[HEALTH_CONDITION]/nh8_el12_dm128_df256_seed41/checkpoint.pth"
```

2. **Number of subjects** - Update based on the health condition:
```bash
# For healthy controls:
--num_subjects 439

# For MCI patients:
--num_subjects 328

# For dementia patients:
--num_subjects 219
```

3. **Output directory** - Update to reflect the health condition:
```bash
--output_dir "./dataset/SYNTH-CAUEEG2-[HEALTH_CONDITION]"
```

Run the data generation for each health condition:
```bash
sbatch generate_data.sh
```

**Note**: You will need to modify the script and submit it separately for each health condition (hc, mci, dementia) that you trained models for.

### 2.5 Organize and Combine Synthetic Data

After generating synthetic data for all health conditions, combine them into one unified dataset.

First, update the paths in `LEAD/combine_synthetic_data.py` on lines 6, 13, 14, and 15:

```python
# Line 6 - Output root path:
output_root = "/home/stud/timlin/bhome/DiffusionEEG/dataset/SYNTH-CAUEEG2"

# Lines 13-15 - Source paths for each health condition:
hc_path = "/home/stud/timlin/bhome/DiffusionEEG/dataset/SYNTH-CAUEEG2-HC"
mci_path = "/home/stud/timlin/bhome/DiffusionEEG/dataset/SYNTH-CAUEEG2-MCI"
dementia_path = "/home/stud/timlin/bhome/DiffusionEEG/dataset/SYNTH-CAUEEG2-dementia"
```

Replace `/home/stud/timlin/bhome/DiffusionEEG` with your actual DiffusionEEG base directory path.

Then run the organization script:
```bash
sbatch LEAD/organize_data.sh
```

### 2.6 Normalize Synthetic Data

The final step is to normalize the synthetic data to match the characteristics of the original dataset.

Update the paths in `LEAD/run_normalize_lead.sh` on lines 178, 179, and 185:

```python
# Lines 178-179 - Dataset paths:
synth_dataset = "/home/stud/timlin/bhome/DiffusionEEG/dataset/SYNTH-CAUEEG2"
reference_data = "/home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG2/Feature"

# Line 185 - Output directory:
output_dir = "/home/stud/timlin/bhome/DiffusionEEG/dataset/SYNTH-CAUEEG2-NORMALIZED"
```

Replace `/home/stud/timlin/bhome/DiffusionEEG` with your actual DiffusionEEG base directory path.

Run the normalization:
```bash
sbatch LEAD/run_normalize_lead.sh
```

**Note**: This script only normalizes the Feature data. You will need to manually copy the Label folder from the non-normalized dataset to the new normalized dataset folder:

```bash
cp -r dataset/SYNTH-CAUEEG2/Label dataset/SYNTH-CAUEEG2-NORMALIZED/
```

**Expected Output**: Upon completion, you will have a normalized synthetic dataset ready for use in `SYNTH-CAUEEG2-NORMALIZED`.

## Step 3: Synthetic Sleep EEG Signal Generation using Latent Diffusion Models

### 3.1 Create LDM Environment

Create the required conda environment for the Latent Diffusion Models:

```bash
sbatch Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/create_env.sh
```

### 3.2 Configure Diffusion Model Settings

The diffusion model configurations and naming are defined in:
```
Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/project/config/config_dm.yaml
```

Before training, update line 10 in the configuration file to set the run directory for your model:
```yaml
run_dir: 'dm_eeg_caueeg2_no_spec'  # Change this to your desired model name
```

### 3.3 Train Pure LDM (No Spectral)

To train the diffusion model without spectral features (dm_no_spec), first ensure the config directory paths are updated for your system, then run:

```bash
sbatch Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/run_train_pure_ldm.sh [CONDITION]
```

Where `[CONDITION]` is one of:
- `hc` - healthy controls only
- `mci` - MCI patients only  
- `dementia` - dementia patients only

**Example usage:**
```bash
# Train on healthy controls
sbatch Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/run_train_pure_ldm.sh hc

# Train on MCI patients
sbatch Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/run_train_pure_ldm.sh mci

# Train on dementia patients
sbatch Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/run_train_pure_ldm.sh dementia
```

**Note**: The script includes automatic requeue functionality for long training jobs and validates the condition parameter before starting training.

### 3.4 Train Pure LDM with Spectral Loss

To train the diffusion model with spectral features, you have two spectral weight options:

#### 3.4.1 Spectral Weight 1E-6

1. Update the run directory in the config file (line 10):
```yaml
run_dir: 'dm_eeg_caueeg2_spec_1e6'  # Change to reflect spectral training
```

2. Uncomment line 59 in `run_train_pure_ldm.sh`:
```bash
#   --spe spectral 
```
Change to:
```bash
    --spe spectral 
```

3. Run the training:
```bash
sbatch Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/run_train_pure_ldm.sh [CONDITION]
```

#### 3.4.2 Spectral Weight 1E-2

1. Update the run directory in the config file (line 10):
```yaml
run_dir: 'dm_eeg_caueeg2_spec_1e2'  # Change to reflect spectral weight
```

2. Modify the spectral weight in `Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/src/train_pure_ldm.py` on line 218:
```python
# Change from:
spectral_weight=1E-6
# To:
spectral_weight=1E-2
```

3. Uncomment line 59 in `run_train_pure_ldm.sh` (same as above):
```bash
    --spe spectral 
```

4. Run the training:
```bash
sbatch Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/run_train_pure_ldm.sh [CONDITION]
```

### 3.5 Generate Synthetic Data from Trained Models

After training the diffusion models, you can generate synthetic EEG data for each health condition.

First, update the paths in `Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/run_generate_dm_eeg.sh` on lines 54-60:

```bash
# Update these paths to match your system and trained models:
--hc_model_path /home/stud/timlin/bhome/DiffusionEEG/Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/project/outputs/dm_eeg_caueeg2_no_spec_label_0/final_model.pth \
--mci_model_path /home/stud/timlin/bhome/DiffusionEEG/Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/project/outputs/dm_eeg_caueeg2_no_spec_label_1/final_model.pth \
--dementia_model_path /home/stud/timlin/bhome/DiffusionEEG/Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/project/outputs/dm_eeg_caueeg2_no_spec_label_2/final_model.pth \
--diffusion_config /home/stud/timlin/bhome/DiffusionEEG/Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/project/config/config_dm.yaml \
--original_label_path /home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG2/Label/label.npy \
--original_data_path /home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG2/Feature \
--output_dir /home/stud/timlin/bhome/DiffusionEEG/dataset/DM_NO_SPEC \
```

**Important**: 
- Replace `/home/stud/timlin/bhome/DiffusionEEG` with your actual DiffusionEEG base directory path
- Update the model path directories (e.g., `dm_eeg_caueeg2_no_spec`) to match the `run_dir` you configured in your training (e.g., `dm_eeg_caueeg2_spec_1e6` if you trained with spectral loss)

Then run the generation script for each condition:
```bash
# Generate healthy control data
sbatch Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/run_generate_dm_eeg.sh hc

# Generate MCI data
sbatch Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/run_generate_dm_eeg.sh mci

# Generate dementia data
sbatch Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/run_generate_dm_eeg.sh dementia
```

## Step 4: Latent Diffusion Model (LDM) Data Generation

### 4.1 Set Up AutoEncoderKL (AEKL) Models

Before generating LDM data, you need to train the AutoEncoderKL models first.

#### 4.1.1 Configure AEKL Settings

Update the run directory in the AEKL config file to reflect your model configuration. For example, for a model without spectral features:

```yaml
run_dir: 'aekl_eeg_4channels_no_spec'  # Change to indicate your model configuration
```

#### 4.1.2 Update AEKL Training Paths

Update the paths in `Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/run_train_aekl.sh` on lines 54 and 55:

```bash
# Line 54:
--path_pre_processed /home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG2 \
# Line 55:
--config_file /home/stud/timlin/bhome/DiffusionEEG/Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/project/config/config_encoder_eeg.yaml \
```

Replace `/home/stud/timlin/bhome/DiffusionEEG` with your actual DiffusionEEG base directory path.

#### 4.1.3 Train AEKL Models

Run the AEKL training for each health condition:

```bash
# Train AEKL for healthy controls
sbatch Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/run_train_aekl.sh hc

# Train AEKL for MCI patients
sbatch Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/run_train_aekl.sh mci

# Train AEKL for dementia patients
sbatch Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/run_train_aekl.sh dementia
```

**Note**: To train AEKL with spectral loss, uncomment lines 58 and 59 in `Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/run_train_aekl.sh`:

```bash
# Change from:
    # --spe spectral \
    # --spectral_cap 10

# To:
    --spe spectral \
    --spectral_cap 10
```

Don't forget to also update the run directory in the config file to reflect the spectral training:
```yaml
run_dir: 'aekl_eeg_4channels_spec'  # Change to indicate spectral features
```

### 4.2 Train Latent Diffusion Model (LDM)

After training the AEKL models, you can now train the actual Latent Diffusion Models.

#### 4.2.1 Configure LDM Settings

Update the run directory in `Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/project/config/config_ldm.yaml` on line 11:

```yaml
run_dir: 'ldm_caueeg2_4ch_116size_no_spec'  # Update to reflect your model configuration
```

#### 4.2.2 Update LDM Training Paths

Update the paths in `Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/run_train_ldm.sh` on lines 57, 58, and 59:

```bash
# Line 57:
--path_pre_processed /home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG2 \
# Line 58:
--best_model_path "/home/stud/timlin/bhome/DiffusionEEG/Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/project/outputs/aekl_eeg_4channels_no_spec_label_${CONDITION}" \
# Line 59:
--autoencoderkl_config_file_path "/home/stud/timlin/bhome/DiffusionEEG/Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/project/config/config_encoder_eeg.yaml" \
```

**Important**: 
- Replace `/home/stud/timlin/bhome/DiffusionEEG` with your actual DiffusionEEG base directory path
- Update the AEKL model path (`aekl_eeg_4channels_no_spec`) to match the run_dir you used when training the AEKL models

#### 4.2.3 Train LDM Models

Run the LDM training for each health condition:

```bash
# Train LDM for healthy controls
sbatch Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/run_train_ldm.sh hc

# Train LDM for MCI patients
sbatch Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/run_train_ldm.sh mci

# Train LDM for dementia patients
sbatch Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/run_train_ldm.sh dementia
```

### 4.3 Generate LDM Synthetic Data

After training both AEKL and LDM models, you can generate synthetic EEG data using the latent diffusion approach.

#### 4.3.1 Update LDM Generation Paths

Update the paths in `Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/run_generate_eeg.sh` on lines 54-64:

```bash
# Update these paths to match your system and trained models:
--hc_model_path /home/stud/timlin/bhome/DiffusionEEG/Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/project/outputs/ldm_caueeg2_4ch_116size_label_hc/final_model.pth \
--mci_model_path /home/stud/timlin/bhome/DiffusionEEG/Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/project/outputs/ldm_caueeg2_4ch_116size_label_mci/final_model.pth \
--dementia_model_path /home/stud/timlin/bhome/DiffusionEEG/Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/project/outputs/ldm_caueeg2_4ch_116size_label_dementia/final_model.pth \
--hc_autoencoder_path /home/stud/timlin/bhome/DiffusionEEG/Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/project/outputs/aekl_eeg_4channels_label_hc/best_model.pth \
--mci_autoencoder_path /home/stud/timlin/bhome/DiffusionEEG/Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/project/outputs/aekl_eeg_4channels_label_mci/best_model.pth \
--dementia_autoencoder_path /home/stud/timlin/bhome/DiffusionEEG/Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/project/outputs/aekl_eeg_4channels_label_dementia/best_model.pth \
--autoencoder_config /home/stud/timlin/bhome/DiffusionEEG/Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/project/config/config_encoder_eeg.yaml \
--diffusion_config /home/stud/timlin/bhome/DiffusionEEG/Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/project/config/config_ldm.yaml \
--original_label_path /home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG2/Label/label.npy \
--original_data_path /home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG2/Feature \
--output_dir /home/stud/timlin/bhome/DiffusionEEG/dataset/LDM_PSD_Normalized_FIX \
```

**Important**: 
- Replace `/home/stud/timlin/bhome/DiffusionEEG` with your actual DiffusionEEG base directory path
- Update the model directory names to match the `run_dir` you configured in your LDM training (e.g., `ldm_caueeg2_4ch_116size` should match your LDM config)
- Update the autoencoder directory names to match the `run_dir` you used for AEKL training (e.g., `aekl_eeg_4channels` should match your AEKL config)

#### 4.3.2 Generate LDM Data

Run the LDM generation for each health condition:

```bash
# Generate healthy control data
sbatch Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/run_generate_eeg.sh hc

# Generate MCI data
sbatch Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/run_generate_eeg.sh mci

# Generate dementia data
sbatch Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/run_generate_eeg.sh dementia
```

## Step 5: Dataset Augmentation and Creation

### 5.1 Create Expanded Datasets

Using the existing `ldm-env` environment from the Sleep folder, you can now create augmented datasets that combine genuine and synthetic data.

**Note**: Continue using the same environment and stay in the `Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models` directory for these operations.

#### 5.1.1 Configure Dataset Paths

Update the paths in `Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/run_augment_dataset.sh` on lines 20 and 21:

```bash
# Line 20 - Genuine dataset path:
--genuine_dataset /home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG2 \
# Line 21 - Synthetic dataset path:
--synthetic_dataset /home/stud/timlin/bhome/DiffusionEEG/dataset/DM_SPEC_MINUS_2 \
```

Replace `/home/stud/timlin/bhome/DiffusionEEG` with your actual DiffusionEEG base directory path, and update the synthetic dataset path to match the dataset you want to use for augmentation.

#### 5.1.2 Run Dataset Augmentation

Execute the dataset augmentation script:

```bash
sbatch Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/run_augment_dataset.sh
```

This script will create multiple augmented datasets with different synthetic data percentages (20%, 40%, 60%, 80%, 100%) while maintaining a stratified test split of 20%.

### 5.2 Create Balanced Datasets

To create balanced datasets where all classes have equal representation, you can use the balance dataset script.

#### 5.2.1 Configure Balance Dataset Paths

Update the paths in `Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/run_balance_dataset.sh` on lines 20, 21, and 22:

```bash
# Line 20 - Genuine dataset path:
--genuine_dataset /home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG2 \
# Line 21 - Synthetic dataset path:
--synthetic_dataset /home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG2_FTSurrogate \
# Line 22 - Output directory path:
--output_dir /home/stud/timlin/bhome/DiffusionEEG/dataset/ftsurrogate_balanced_datasets \
```

Also update the dataset type on line 26:
```bash
--dataset_type augmented  # Change to either "augmented" or "synthetic"
```

Replace `/home/stud/timlin/bhome/DiffusionEEG` with your actual DiffusionEEG base directory path.

#### 5.2.2 Run Dataset Balancing

Execute the dataset balancing script:

```bash
sbatch Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/run_balance_dataset.sh
```

This script balances the dataset to the maximum class size while maintaining a stratified test split of 20%.

### 5.3 Create Discrimination Datasets

To create datasets for discriminating between genuine and synthetic/augmented data, you can use the discrimination dataset script.

#### 5.3.1 Configure Discrimination Dataset Paths

Update the paths in `Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/run_discrimination_datasets.sh` on lines 20, 21, 22, and 23:

```bash
# Line 20 - Genuine dataset path:
--genuine_dataset /home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG2 \
# Line 21 - Comparison dataset path:
--comparison_dataset /home/stud/timlin/bhome/DiffusionEEG/dataset/DM_NO_SPEC \
# Line 22 - Comparison type:
--comparison_type synthetic \  # Change to either "synthetic" or "augmented"
# Line 23 - Output directory path:
--output_dir /home/stud/timlin/bhome/DiffusionEEG/dataset/dm_no_spec_discrimination_synthetic \
```

Replace `/home/stud/timlin/bhome/DiffusionEEG` with your actual DiffusionEEG base directory path.

#### 5.3.2 Run Discrimination Dataset Creation

Execute the discrimination dataset creation script:

```bash
sbatch Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/run_discrimination_datasets.sh
```

This script creates datasets for binary classification between genuine and synthetic/augmented data with a 20% validation split.