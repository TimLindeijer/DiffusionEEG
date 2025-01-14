Google Disk: https://drive.google.com/drive/folders/1qvIyEF0INjIYH-uF4PS9KQdnCx9gnq0H
Some Steps i did:
uenv miniconda3-py311
conda create --name DiffusionEEG python=3.11
conda activate DiffusionEEG
cd eeg_harmonization
pip install -r requirements.txt 
OBS! caueeg_bids.rar needs a dataset_description.json to work, fill the json file with this:
{
  "Name": "caueeg_dataset",
  "BIDSVersion": "1.0.2"
}
