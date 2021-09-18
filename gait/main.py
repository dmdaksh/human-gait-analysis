from gait.read_data import ReadData
from gait.config import Config, Flags
# from gait.gait_dataset import GaitDataset
# from gait.read_data import ReadData
from gait import log
from dataclasses import asdict
import argparse
import time
import os
# from gait.train_tpu import run
from gait.train_gpu import kfold_run

if __name__ == '__main__':
    config_dict = asdict(Config())
    parser = argparse.ArgumentParser()
    parser.add_argument('--exclude_surface', type = str, nargs = '+', help = 'To exclude surfaces while loading data', \
        choices = config_dict['SURFACES'])
    parser.add_argument('--exclude_loc', type = str, nargs = '+', help = 'To exclude sensor locations while loading data', \
        choices = config_dict['LOCS'])

    args = parser.parse_args()
    exclude_surface = set(args.exclude_surface) if args.exclude_surface else set()
    exclude_loc = set(args.exclude_loc) if args.exclude_loc else set()

    logger = log.get_logger(__name__)

    logger.info('human-gait-analysis')
    
    logger.info('Included surfaces: {}'.format(config_dict['SURFACES']-exclude_surface))
    logger.info('Included sensor locs: {}'.format(config_dict['LOCS']-exclude_loc))


    FLAGS = asdict(Flags(
        SURFACES=config_dict['SURFACES']-exclude_surface, 
        LOCS=config_dict['LOCS']-exclude_loc))
    

    if not os.path.isfile(config_dict['PREPROCESSED_ARR']):
        ReadData()._dump_processed_data(FLAGS, config_dict['PREPROCESSED_ARR'])
    
    kfold_run(FLAGS, config_dict)
