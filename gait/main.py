import argparse
from dataclasses import asdict
import os

from gait.config import Config, Flags
from gait.read_data import ReadData
from gait.train_gpu import kfold_run

if __name__ == '__main__':
    config_dict = asdict(Config())
    parser = argparse.ArgumentParser()
    parser.add_argument('--exclude_surface',
                        type=str,
                        nargs='+',
                        help='To exclude surfaces while loading data',
                        choices=config_dict['SURFACES'])
    parser.add_argument('--exclude_loc',
                        type=str,
                        nargs='+',
                        help='To exclude sensor locations while loading data',
                        choices=config_dict['LOCS'])
    parser.add_argument('--pause_instance',
                        help='Want to pause instance after training',
                        action='store_true',
                        dest='pause_instance')
    parser.add_argument('--no_pause_instance',
                        help='Want to pause instance after training',
                        action='store_false',
                        dest='pause_instance')

    args = parser.parse_args()
    exclude_surface = set(
        args.exclude_surface) if args.exclude_surface else set()
    exclude_loc = set(args.exclude_loc) if args.exclude_loc else set()
    pause_instance = args.pause_instance

    print('human-gait-analysis')

    print('Included surfaces: {}'.format(config_dict['SURFACES'] -
                                         exclude_surface))
    print('Included sensor locs: {}'.format(config_dict['LOCS'] - exclude_loc))
    print('Pause Instance: {}'.format(pause_instance))

    FLAGS = asdict(
        Flags(SURFACES=config_dict['SURFACES'] - exclude_surface,
              LOCS=config_dict['LOCS'] - exclude_loc))

    if not os.path.isfile(config_dict['PREPROCESSED_ARR']):
        ReadData()._dump_processed_data(FLAGS, config_dict['PREPROCESSED_ARR'])

    kfold_run(FLAGS, config_dict)

    # pause instance
    # if pause_instance:
    #     from jarviscloud import jarviscloud
    #     jarviscloud.pause()
