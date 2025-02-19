import os
import mne
import numpy as np
import pandas as pd
from scipy.signal import resample
# import h5py
# import mat73
import warnings
warnings.filterwarnings("ignore")


path = 'dataset/CAUEEG/Feature'

for file in os.listdir(path):
    sub_path = os.path.join(path, file)
    print(np.load(sub_path).shape)