#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=03:15:00
#SBATCH --job-name=LEAD_10e
#SBATCH --output=outputs/LEAD_10e.out
 
# Activate environment
uenv verbose cuda-12.1.0 cudnn-12.x-9.0.0
uenv miniconda3-py38
conda activate LEADEEG

export CUDA_VISIBLE_DEVICES=0,1,2,3

python -u LEAD/LEAD/run.py --method LEAD --task_name supervised --is_training 1 --root_path ./dataset/ --model_id S-1-V-CAUEEG-Sup --model LEAD --data MultiDatasets \
--training_datasets CAUEEG \
--testing_datasets CAUEEG \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 10 --patience 15
# pip uninstall torch torchvision torchaudio
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# python -c "import torch
# print(torch.__version__);
# print(torch.version.cuda);
# print(torch.backends.cudnn.version());
# print(torch.cuda.is_available());
# print(torch.cuda.device_count());
# print(torch.cuda.get_device_name(0));
# "