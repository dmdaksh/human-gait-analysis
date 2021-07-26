from src import log
from src.config import INPUT_DATA, SUBJECTS, SURFACE_TRIALS, SENSOR_LOCS
import os
import glob
import numpy as np
import re

logger = log.get_logger(__name__)

class GaitDataset:
    def __init__(self, surfaces, locs) -> None:
        self.surfaces = surfaces    # included surfaces
        self.locs = locs    # included locs
        self.pattern = '*.csv'
        # self.data = self._load_data()

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def _get_filenames(self, subject):
        filenames = set()
        all_filenames = set(glob.glob(os.path.join(
            INPUT_DATA, str(subject), self.pattern)))

        for filename in all_filenames:
            # if filename: '../gait_data/input_data/1/6-000_00B43295.txt.csv', then split_filename: ['1', '000', '00B4329B', 'txt', 'csv']
            split_filename =  re.split('-|_|\.', re.split('/', filename)[-1])
            for surface in self.surfaces:
                if(int(split_filename[0]) in SURFACE_TRIALS[surface]):
                    for loc in self.locs:
                        if(split_filename[2][-2:] == SENSOR_LOCS[loc]):
                            filenames.add(filename)
        return filenames
                                

    def _load_data(self):
        for subject in list(range(1, SUBJECTS+1)):
            filenames = self._get_filenames(subject)





