from gait.read_data import ReadData
from gait.config import Config, Flags
# from gait.gait_dataset import GaitDataset
# from gait.read_data import ReadData
from gait import log
from dataclasses import asdict
import argparse
import time
# from gait.train import run
from gait.train_gpu import run

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

    # logger.debug('Instantiating GaitDataset class')
    # start = time.perf_counter()
    # dataset = GaitDataset(config_dict['SURFACES']-exclude_surface, config_dict['LOCS']-exclude_loc)
    # logger.info(f'Time taken: {time.perf_counter() - start}')
    # logger.info('No. of samples: {}'.format(len(dataset)))
    # logger.info(dataset[100]['surface_label'])
    # # logger.info(dataset._get_filenames(1))
    # logger.info('sleeping for 10 seconds')
    # time.sleep(10)
    # logger.info('done sleeping')

    # # getting flag dicts
    FLAGS = asdict(Flags(
        SURFACES=config_dict['SURFACES']-exclude_surface, 
        LOCS=config_dict['LOCS']-exclude_loc))
    
    # running model
    # data = ReadData()._init_data(FLAGS)
    # print(*data)
    run(FLAGS)
    # print(acc.shape, gyr.shape, mag.shape, targets.shape)
    # logger.debug('acc_samples.shape: {}, gyr_samples.shape: {}, mag_samples.shape: {}, surface_labels.shape: {}'.format(
    #     samples['acc_samples'].shape,
    #     samples['gyr_samples'].shape,
    #     samples['mag_samples'].shape,
    #     samples['surface_labels'].shape
    # ))
    # print(samples['surface_labels'].dtype)
    # ReadData().dump_processed_data(FLAGS, 'preprocessed_data.pkl')


