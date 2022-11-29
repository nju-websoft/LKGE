import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import time
import random
from prettytable import PrettyTable
from torch.nn.init import xavier_normal_
from torch.nn import Parameter
import numpy as np
from copy import deepcopy
import sys, os
from torch.backends import cudnn


def get_param(shape):
    '''create learnable parameters'''
    param = Parameter(torch.Tensor(*shape)).double()
    xavier_normal_(param.data)
    return param


def same_seeds(seed):
    '''Set seed for reproduction'''
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_fact(path):
    '''
    Load (sub, rel, obj) from file 'path'.
    :param path: xxx.txt
    :return: fact list: [(s, r, o)]
    '''
    facts = []
    with open(path, 'r') as f:
        for line in f:
            line = line.split()
            s, r, o = line[0], line[1], line[2]
            facts.append((s, r, o))
    return facts


def build_edge_index(s, o):
    '''build edge_index using subject and object entity'''
    index = [s + o, o + s]
    return torch.LongTensor(index)
