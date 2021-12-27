import torch
from torch.utils.data import Dataset

from gait import log

logger = log.get_logger(__name__)


class GaitDataset(Dataset):
    def __init__(self, acc, gyr, mag, targets, device=None) -> None:
        self.acc = torch.tensor(acc, dtype=torch.float32, device=device)
        self.gyr = torch.tensor(gyr, dtype=torch.float32, device=device)
        self.mag = torch.tensor(mag, dtype=torch.float32, device=device)
        self.targets = torch.tensor(targets, dtype=torch.long, device=device)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (self.acc[idx, :], self.gyr[idx, :], self.mag[idx, :],
                self.targets[idx])


# class GaitDataset(Dataset):
#     def __init__(self, surfaces, locs, window_size=3, overlap=1) -> None:
#         self.config_dict = asdict(Config())
#         self.surfaces = surfaces  # included surfaces
#         self.locs = locs  # included locs
#         self.pattern = '*.csv'
#         self.cols = [
#             'Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'Mag_X',
#             'Mag_Y', 'Mag_Z'
#         ]
#         self.window_size = window_size
#         self.overlap = overlap
#         self.sampling_rate = 100
#         self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#         self.samples = {}
#         self.label_encoder = LabelEncoder()
#         self.onehot_encoder = OneHotEncoder()
#         self._init_data()
#
#     def __len__(self):
#         # len = 133728 (2^5, 3, 7, 199)
#         return len(self.samples['surface_labels'])
#         # pass
#
#     def __getitem__(self, idx):
#         return {
#             'acc_sample':
#             torch.from_numpy(self.samples['acc_samples'][idx, :]),
#             'gyr_sample':
#             torch.from_numpy(self.samples['gyr_samples'][idx, :]),
#             'mag_sample':
#             torch.from_numpy(self.samples['mag_samples'][idx, :]),
#             'surface_label':
#             torch.from_numpy(self.samples['surface_labels'][idx])
#         }
#         # pass
#
#     def _get_filenames(self, subject):
#         logger.debug('Inside _get_filenames')
#         filenames = set()
#         all_filenames = set(
#             glob.glob(
#                 os.path.join(self.config_dict['INPUT_DATA'], str(subject),
#                              self.pattern)))
#
#         for filename in all_filenames:
#             # if filename: '../gait_data/input_data/1/6-000_00B43295.txt.csv', then split_filename: ['1', '000', '00B4329B', 'txt', 'csv']
#             split_filename = re.split('-|_|\.', re.split('/', filename)[-1])
#             for surface in self.surfaces:
#                 if (int(split_filename[0])
#                         in self.config_dict['SURFACE_TRIALS'][surface]):
#                     for loc in self.locs:
#                         if (split_filename[2][-2:] ==
#                                 self.config_dict['SENSOR_LOCS'][loc]):
#                             filenames.add(filename)
#         logger.debug('Exiting _get_filenames')
#         return filenames
#
#     def _init_data(self):
#         logger.debug('Inside _init_data')
#
#         trial_surfaces = {
#             v_k: k
#             for k, v in self.config_dict['SURFACE_TRIALS'].items() for v_k in v
#         }
#
#         window_length = int(self.window_size * self.sampling_rate)
#         overlap_length = int(self.overlap * self.sampling_rate)
#
#         acc_samples, gyr_samples, mag_samples = [], [], []
#         surface_labels = []
#
#         for subject in tqdm(
#                 range(1, 1 + 1),
#                 desc='Loading Subjects'):  #self.config_dict['SUBJECTS']
#             filenames = self._get_filenames(subject)
#             for filename in filenames:
#                 df_arr = pd.read_csv(filename,
#                                      usecols=self.cols,
#                                      skipinitialspace=True,
#                                      engine='c').values
#
#                 # It gets the surface number from the filename and trial_surfaces dict returns surface code
#                 surface = trial_surfaces[int(
#                     re.split('-|_|\.',
#                              re.split('/', filename)[-1])[0])]
#                 start, end = 0, len(df_arr)
#                 while end > start + window_length:
#                     acc_samples.append(df_arr[start:start + window_length, :3])
#                     gyr_samples.append(df_arr[start:start + window_length,
#                                               3:6])
#                     mag_samples.append(df_arr[start:start + window_length,
#                                               6:9])
#                     surface_labels.append(surface)
#                     start += overlap_length
#
#         surface_label = self.onehot_encoder.fit_transform(
#             self.label_encoder.fit_transform(surface_labels).reshape(
#                 -1, 1)).toarray()
#
#         self.samples['acc_samples'] = np.array(acc_samples,
#                                                dtype=np.float).transpose(
#                                                    0, 2, 1)
#         self.samples['gyr_samples'] = np.asarray(gyr_samples,
#                                                  dtype=np.float).transpose(
#                                                      0, 2, 1)
#         self.samples['mag_samples'] = np.asarray(mag_samples,
#                                                  dtype=np.float).transpose(
#                                                      0, 2, 1)
#         self.samples['surface_labels'] = np.asarray(surface_label,
#                                                     dtype=np.float)
#
#         logger.debug(
#             'acc_samples.shape: {}, gyr_samples.shape: {}, mag_samples.shape: {}, surface_labels.shape: {}'
#             .format(self.samples['acc_samples'].shape,
#                     self.samples['gyr_samples'].shape,
#                     self.samples['mag_samples'].shape,
#                     self.samples['surface_labels'].shape))
#
#         logger.debug('Exit _init_date')
