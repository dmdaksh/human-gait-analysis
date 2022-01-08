from dataclasses import dataclass, field
import logging


@dataclass
class Config:
    LOG_LEVELS: dict = field(
        default_factory=lambda: {
            'critical': logging.CRITICAL,
            'error': logging.ERROR,
            'warning': logging.WARNING,
            'info': logging.INFO,
            'debug': logging.DEBUG
        })
    MAT_FILE: str = field(default='../gait_data/22008945')
    INPUT_DATA: str = field(default='../input_data/')
    PREPROCESSED_ARR: str = field(default='data/preprocessed_arr.pkl')
    SURFACE_TRIALS: dict = field(
        default_factory=lambda: {
            'CALIB': [1, 2, 3],  # Calibration
            'FE': [4, 5, 6, 7, 8, 9],  # Flat even
            'CS': [10, 11, 12, 13, 14, 15],  # Cobble stone
            'StrU': [16, 18, 20, 22, 24, 26],  # Upstairs
            'StrD': [17, 19, 21, 23, 25, 27],  # Downstairs
            'SlpU': [28, 30, 32, 34, 36, 38],  # Slope up
            'SlpD': [29, 31, 33, 35, 37, 39],  # Slope down
            'BnkL': [40, 42, 44, 46, 48, 50],  # Bank left
            'BnkR': [41, 43, 45, 47, 49, 51],  # Bank right
            'GR': [52, 53, 54, 55, 56, 57]  # Grass
        })
    SURFACES: set = field(default_factory=lambda: {
        'CALIB', 'FE', 'CS', 'StrU', 'StrD', 'SlpU', 'SlpD', 'BnkL', 'BnkR',
        'GR'
    })
    SENSOR_LOCS: dict = field(
        default_factory=lambda: {
            'Trunk': 'CC',
            'Wrist': '95',
            'RightThigh': '93',
            'LeftThigh': '8B',
            'RightShank': '9B',
            'LeftShank': 'B6'
        })
    LOCS: set = field(default_factory=lambda: {
        'Trunk', 'Wrist', 'RightThigh', 'LeftThigh', 'RightShank', 'LeftShank'
    })
    SUBJECTS: int = field(default=30)


@dataclass
class Flags:
    SURFACES: dict
    LOCS: dict
    SAMPLING_RATE: int = field(default=100)
    WINDOW_SIZE: int = field(default=3)
    OVERLAP: int = field(default=1)
    MODE: str = field(default='train')
    BATCH_SIZE: int = field(default=2048)  # rtx5000 tensorcores = 384
    LEARNING_RATE: float = field(default=1e-3)
    MOMENTUM: float = field(default=0.5)
    EPOCHS: int = field(default=70)
    LOG_STEPS: int = field(default=10)
    METRICS_DEBUG: bool = field(default=False)
    WORLD_SIZE: int = field(default=8)
    SCHEDULER: bool = field(default=True)
    WEIGHT_DECAY: float = field(default=0.001)
