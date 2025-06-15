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