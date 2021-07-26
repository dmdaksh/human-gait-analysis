import logging


LOG_LEVELS = {
      'critical':   logging.CRITICAL
    , 'error':      logging.ERROR
    , 'warning':    logging.WARNING
    , 'info':       logging.INFO
    , 'debug':      logging.DEBUG
}
MAT_FILE = '../gait_data/22008945'
INPUT_DATA = '../gait_data/input_data/'
SURFACE_TRIALS = {
      'CALIB':  [1, 2, 3]                   # Calibration
    , 'FE':     [4, 5, 6, 7, 8, 9]          # Flat even
    , 'CS':     [10, 11, 12, 13, 14, 15]    # Cobble stone
    , 'StrU':   [16, 18, 20, 22, 24, 26]    # Upstairs
    , 'StrD':   [17, 19, 21, 23, 25, 27]    # Downstairs
    , 'SlpU':   [28, 30, 32, 34, 36, 38]    # Slope up
    , 'SlpD':   [29, 31, 33, 35, 37, 39]    # Slope down
    , 'BnkL':   [40, 42, 44, 46, 48, 50]    # Bank left
    , 'BnkR':   [41, 43, 45, 47, 49, 51]    # Bank right
    , 'GR':     [52, 53, 54, 55, 56, 57]    # Grass
}
SURFACES = {'CALIB', 'FE', 'CS', 'StrU', 'StrD', 'SlpU', 'SlpD', 'BnkL', 'BnkR', 'GR'}
SENSOR_LOCS = {
      'Trunk':          'CC'
    , 'Wrist':          '95'
    , 'RightThigh':     '93'
    , 'LeftThigh':      '8B'
    , 'RightShank':     '9B'
    , 'LeftShank':      'B6'
}
LOCS = {'Trunk', 'Wrist', 'RightThigh', 'LeftThigh', 'RightShank', 'LeftShank'}
SUBJECTS = 30