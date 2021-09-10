from gait import log
from gait.config import Config, Flags
from dataclasses import asdict
import os
import glob
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler, StandardScaler


logger = log.get_logger(__name__)


class ReadData:
    def __init__(self):
        self.config_dict = asdict(Config())

        self.pattern = '*.csv'
        self.cols = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X',
                    'Gyr_Y', 'Gyr_Z', 'Mag_X', 'Mag_Y', 'Mag_Z']
        self.label_encoder = LabelEncoder()
        self.onehot_encoder = OneHotEncoder()
        
    def _get_filenames(self, subject, FLAGS):
        logger.debug('Inside _get_filenames')
        filenames = set()
        all_filenames = set(glob.glob(os.path.join(
            self.config_dict['INPUT_DATA'], str(subject), self.pattern)))

        for filename in all_filenames:
            # if filename: '../gait_data/input_data/1/6-000_00B43295.txt.csv', then split_filename: ['1', '000', '00B4329B', 'txt', 'csv']
            split_filename = re.split('-|_|\.', re.split('/', filename)[-1])
            for surface in FLAGS['SURFACES']:
                if(int(split_filename[0]) in self.config_dict['SURFACE_TRIALS'][surface]):
                    for loc in FLAGS['LOCS']:
                        if(split_filename[2][-2:] == self.config_dict['SENSOR_LOCS'][loc]):
                            filenames.add(filename)
        logger.debug('Exiting _get_filenames')
        return filenames

    def mean_fn(self, x):
        return x.mean()

    def _init_data(self, FLAGS):
        logger.debug('Inside _init_data')

        trial_surfaces = {v_k:k for k, v in self.config_dict['SURFACE_TRIALS'].items() for v_k in v}

        window_length = int(FLAGS['WINDOW_SIZE'] * FLAGS['SAMPLING_RATE'])
        overlap_length = int(FLAGS['OVERLAP'] * FLAGS['SAMPLING_RATE'])

        acc_samples, gyr_samples, mag_samples = [], [], []
        surface_labels = []

        for subject in tqdm(range(1, self.config_dict['SUBJECTS']+1), desc='Loading Subjects'): #self.config_dict['SUBJECTS']
            filenames = self._get_filenames(subject, FLAGS)
            for filename in filenames:
                df_arr = pd.read_csv(filename, usecols=self.cols,
                                        skipinitialspace=True, engine='c').values

                # skipping nan files
                if(np.isnan(df_arr).sum()):
                    print(np.isnan(df_arr).sum())
                    continue
                
                df_arr = StandardScaler().fit_transform(df_arr)

                # It gets the surface number from the filename and trial_surfaces dict returns surface code
                surface = trial_surfaces[int(re.split('-|_|\.', re.split('/', filename)[-1])[0])]
                start, end = 0, len(df_arr)
                while end > start + window_length:
                    acc_samples.append(df_arr[start:start+window_length, :3])
                    gyr_samples.append(df_arr[start:start+window_length, 3:6])
                    mag_samples.append(df_arr[start:start+window_length, 6:9])
                    surface_labels.append(surface)
                    start += overlap_length


        # surface_labels = self.onehot_encoder.fit_transform(self.label_encoder.fit_transform(surface_labels).reshape(-1,1)).toarray()
        surface_labels = self.label_encoder.fit_transform(surface_labels).reshape(-1,1)

        return (
            np.asarray(acc_samples, dtype=np.float).transpose(0,2,1),
            np.asarray(gyr_samples, dtype=np.float).transpose(0,2,1), 
            np.asarray(mag_samples, dtype=np.float).transpose(0,2,1),
            np.asarray(surface_labels, dtype=np.long).reshape(-1)
        )

