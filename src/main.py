from src import gait_dataset
from src.config import SURFACES, LOCS
from src.gait_dataset import GaitDataset
from src import log
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exclude_surface', type = str, nargs = '+', help = 'To exclude surfaces while loading data', \
        choices = SURFACES)
    parser.add_argument('--exclude_loc', type = str, nargs = '+', help = 'To exclude sensor locations while loading data', \
        choices = LOCS)

    args = parser.parse_args()
    exclude_surface = set(args.exclude_surface) if args.exclude_surface else set()
    exclude_loc = set(args.exclude_loc) if args.exclude_loc else set()

    logger = log.get_logger(__name__)

    logger.info('human-gait-analysis')
    
    logger.info('Included surfaces: {}'.format(SURFACES-exclude_surface))
    logger.info('Included sensor locs: {}'.format(LOCS-exclude_loc))

    logger.debug('Instantiating GaitDataset class')
    dataset = GaitDataset(SURFACES-exclude_surface, LOCS-exclude_loc)
    logger.info('No. of samples: {}'.format(len(dataset)))    
