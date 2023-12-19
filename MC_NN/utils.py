import torch
import config
import random
import numpy as np


def random_seed(random_seed = config.RANDOM_SEED):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
