from src import config
from src import dataset_prep
from src import log
import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exclude_surface', type = str, nargs = '+', help = 'To exclude surfaces while loading data', \
        choices = ['CALIB', 'FE', 'CS', 'StrU', 'StrD', 'SlpU', 'SlpD', 'BnkL', 'BnkR', 'GR'])
    
    args = parser.parse_args()

    logger = log.get_logger(__name__)

    logger.info('human-gait-analysis')
    
    
