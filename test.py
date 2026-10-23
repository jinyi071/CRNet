import argparse
import os

import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import PreFetcher
from utils import Tester, init_device, init_model, logger


def parse_args():
    parser = argparse.ArgumentParser(description='CRNet test-only evaluation')
    parser.add_argument('--pretrained', type=str, required=True,
                        help='checkpoint containing weights and test configuration')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU id to use')
    parser.add_argument('--cpu', action='store_true',
                        help='run evaluation on CPU')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed')
    return parser.parse_args()


def load_checkpoint_config(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Checkpoint does not exist: {path}')

    checkpoint = torch.load(
        path,
        map_location=torch.device('cpu'),
        weights_only=False)
    config = checkpoint.get('config')
    required = {'data_dir', 'scenario', 'cr', 'batch_size', 'workers'}
    missing = required - set(config or {})
    if missing:
        fields = ', '.join(sorted(missing))
        raise ValueError(
            f'Checkpoint does not contain required test configuration: {fields}. '
            'Use a checkpoint produced by the updated training code.')
    return config


def create_test_loader(root, scenario, batch_size, workers, pin_memory):
    if not os.path.isdir(root):
        raise FileNotFoundError(f'Dataset directory does not exist: {root}')

    test_path = os.path.join(root, f'DATA_Htest{scenario}.mat')
    raw_path = os.path.join(root, f'DATA_HtestF{scenario}_all.mat')
    for path in (test_path, raw_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Required test file does not exist: {path}')

    data_test = sio.loadmat(test_path)['HT']
    data_test = torch.tensor(data_test, dtype=torch.float32).view(
        data_test.shape[0], 2, 32, 32)

    raw_test = sio.loadmat(raw_path)['HF_all']
    if raw_test.shape[0] != data_test.shape[0]:
        raise ValueError('HT and HF_all contain different numbers of samples.')
    raw_test = torch.tensor(raw_test, dtype=torch.complex64).view(
        raw_test.shape[0], 32, 125)
    raw_test = torch.view_as_real(raw_test)

    test_loader = DataLoader(
        TensorDataset(data_test, raw_test),
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=pin_memory,
        shuffle=False)

    if pin_memory:
        test_loader = PreFetcher(test_loader)
    return test_loader


def main():
    args = parse_args()
    config = load_checkpoint_config(args.pretrained)
    args.cr = config['cr']
    args.seed = config.get('seed')
    logger.info('=> PyTorch Version: {}'.format(torch.__version__))

    device, pin_memory = init_device(
        seed=args.seed,
        cpu=args.cpu,
        gpu=args.gpu,
        affinity=None)
    test_loader = create_test_loader(
        root=config['data_dir'],
        scenario=config['scenario'],
        batch_size=config['batch_size'],
        workers=config['workers'],
        pin_memory=pin_memory)

    model = init_model(args)
    model.to(device)
    criterion = nn.MSELoss().to(device)

    loss, rho, nmse = Tester(model, device, criterion)(test_loader)
    print(f'Final test loss: {loss:.3e}'
          f'\nFinal test rho:  {rho:.3e}'
          f'\nFinal test NMSE: {nmse:.3e}')


if __name__ == '__main__':
    main()
