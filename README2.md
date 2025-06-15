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