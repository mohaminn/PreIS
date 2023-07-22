import rita_configuration, rita_modeling
import torch
import config
import random
import numpy as np


def init_rita():

    modelPath = config.RITA_DIR + "/pytorch_model.bin"
    ritaConfig = rita_configuration.RITAConfig()
    model = model = rita_modeling.RITAModelForCausalLM(ritaConfig)
    model.load_state_dict(torch.load(modelPath))
    return model.transformer

def SDA(ids, targets, augment_ids, augment_targets):

    for i in range(len(targets)):

        temp = augment_targets == targets[i]
        temp = augment_ids[temp, :]
        temp = temp[torch.randint(0,len(temp), (1,))]
        idslen = torch.min((ids[i].eq(2) * 1).argmax(-1))
        templen = torch.min((temp.eq(2) * 1).argmax(-1))
        minlen = min(templen, idslen) -1
        auglen = int(config.GAMMA_G * minlen)
        rand_pos = torch.randint(0, (minlen - auglen), (1, ))[0]
        ids[i, rand_pos : rand_pos + auglen] = temp[0][rand_pos : rand_pos + auglen]

        temp = augment_targets == targets[i]
        temp = augment_ids[temp, :]
        temp = temp[torch.randint(0,len(temp), (1,))]
        templen = torch.min((temp.eq(2) * 1).argmax(-1))
        minlen = min(templen, idslen) -1
        rand_pos = torch.randint(0, minlen, (int(config.GAMMA_L * len(ids[i])), ))
        ids[i, rand_pos] = temp[0][rand_pos]

    return ids

def random_seed(random_seed = config.RANDOM_SEED):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)