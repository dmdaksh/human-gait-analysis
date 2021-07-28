from src import log
from src.config import INPUT_DATA, SUBJECTS, SURFACE_TRIALS, SENSOR_LOCS
import os
import glob
import numpy as np
import pandas as pd
import re
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer


logger = log.get_logger(__name__)


class GaitDataset(Dataset):
    def __init__(self, surfaces, locs, window_size=3, overlap=1) -> None:
        self.surfaces = surfaces    # included surfaces
        self.locs = locs    # included locs
        self.pattern = '*.csv'
        self.cols = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X',
                     'Gyr_Y', 'Gyr_Z', 'Mag_X', 'Mag_Y', 'Mag_Z']
        self.window_size = window_size
        self.overlap = overlap
        self.sampling_rate = 100
        self.samples = {}
        self._init_data()

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
        # if torch.is_tensor(idx):
        #     idx = idx.to_list()

    def _get_filenames(self, subject):
        filenames = set()
        all_filenames = set(glob.glob(os.path.join(
            INPUT_DATA, str(subject), self.pattern)))

        for filename in all_filenames:
            # if filename: '../gait_data/input_data/1/6-000_00B43295.txt.csv', then split_filename: ['1', '000', '00B4329B', 'txt', 'csv']
            split_filename = re.split('-|_|\.', re.split('/', filename)[-1])
            for surface in self.surfaces:
                if(int(split_filename[0]) in SURFACE_TRIALS[surface]):
                    for loc in self.locs:
                        if(split_filename[2][-2:] == SENSOR_LOCS[loc]):
                            filenames.add(filename)
        return filenames

    def _init_data(self):
        logger.debug('Inside _init_data method')

        window_length = self.window_size * self.sampling_rate
        overlap_length =  self.overlap * self.sampling_rate

        # acc_data = np.empty((1, window_length, 3))
        # gyr_data = np.empty((1, window_length, 3))
        # mag_data = np.empty((1, window_length, 3))
        acc_samples, gyr_samples, mag_samples, labels = [], [], [], []
        surface_label = []

        for subject in tqdm(range(1, SUBJECTS+1), desc='Subjects'):
            filenames = self._get_filenames(subject)
            for filename in filenames:
                print(filename)
                acc_samples, gyr_samples, mag_samples, labels = [], [], [], []
                df_arr = pd.read_csv(filename, usecols=self.cols,
                                     skipinitialspace=True, engine='c').values
                
                start, end = 0, len(df_arr)
                while end > start + window_length:
                    acc_samples.append(df_arr[start:start+window_length, :3])
                    gyr_samples.append(df_arr[start:start+window_length, 3:6])
                    mag_samples.append(df_arr[start:start+window_length, 6:9])
                    start += overlap_length
                
                # acc_data = np.vstack((acc_data, acc_samples))
                # gyr_data = np.vstack((gyr_data, gyr_samples))
                # mag_data = np.vstack((mag_data, mag_samples))

        self.samples['acc_samples'] = np.asarray(acc_samples, dtype = np.float32)
        self.samples['gyr_samples'] = np.asarray(gyr_samples, dtype = np.float32)
        self.samples['mag_samples'] = np.asarray(mag_samples, dtype = np.float32)
        


    def _labelencode_onehotencode(self):
        pass

    def _inverse_labelencode_onehotencode(self):
        pass

    def _scale_date(self):
        pass
