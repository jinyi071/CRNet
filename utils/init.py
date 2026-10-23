import os
import random
import numpy as np
import thop
import torch

from models import crnet
from utils import logger, line_seg

__all__ = ["seed_everything", "init_device", "init_model"]


def seed_everything(seed=92304):
    if seed is None:
        seed = 92304

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    logger.info(f'=> Random seed: {seed}')


def init_device(seed=92304, cpu=None, gpu=None, affinity=None):
    # set the CPU affinity
    if affinity is not None:
        os.system(f'taskset -p {affinity} {os.getpid()}')

    # Set the GPU id you choose
    if gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    seed_everything(seed)

    # Env setup
    if not cpu and torch.cuda.is_available():
        device = torch.device('cuda')
        pin_memory = True
        logger.info("Running on GPU%d" % (gpu if gpu else 0))
    else:
        pin_memory = False
        device = torch.device('cpu')
        logger.info("Running on CPU")

    return device, pin_memory


def init_model(args):
    # Model loading
    model = crnet(reduction=args.cr)

    if args.pretrained is not None:
        assert os.path.isfile(args.pretrained)
        state_dict = torch.load(args.pretrained,
                                map_location=torch.device('cpu'),
                                weights_only=False)['state_dict']
        model.load_state_dict(state_dict)
        logger.info("pretrained model loaded from {}".format(args.pretrained))

    # Model flops and params counting
    image = torch.randn([1, 2, 32, 32])
    flops, params = thop.profile(model, inputs=(image,), verbose=False)
    flops, params = thop.clever_format([flops, params], "%.3f")

    # Model info logging
    logger.info(f'=> Model Name: CRNet [pretrained: {args.pretrained}]')
    logger.info(f'=> Model Config: compression ratio=1/{args.cr}')
    logger.info(f'=> Model Flops: {flops}')
    logger.info(f'=> Model Params Num: {params}\n')
    logger.info(f'{line_seg}\n{model}\n{line_seg}\n')

    return model
