import random

import numpy as np
from itertools import product
# from PIL import Image

PERI_SIDE_ONE_RANGE = (1.0, 20.0)
PERI_SIDE_TWO_RANGE = (1.0, 20.0)
PERI_SIDE_THREE_RANGE = (1.0, 20.0)

int_to_peri_list = {
    1   : 'single',
    2   : 'pairs',
    3   : 'triplets'
}

def make_single_prod(side_one_range, num_splices):
    singles = list(np.linspace(side_one_range[0], side_one_range[1], num=num_splices))
    
    # singles = [(x, None, None) for x in singles]
    
    return singles

def make_pair_prod(side_one_range, side_two_range, num_splices):
    side_one = np.linspace(side_one_range[0], side_one_range[1], num=num_splices)
    side_two = np.linspace(side_two_range[0], side_two_range[1], num=num_splices)
    
    # Generate all possible pairwise combinations
    pairs = list(product(side_one, side_two))
    # pairs = [(x, y, None) for x, y in pairs]
    
    return pairs


def make_triplet_prod(side_one_range, side_two_range, side_three_range, num_splices):
    side_one = np.linspace(side_one_range[0], side_one_range[1], num=num_splices)
    side_two = np.linspace(side_two_range[0], side_two_range[1], num=num_splices)
    side_three = np.linspace(side_three_range[0], side_three_range[1], num=num_splices)
    
    # Generate all possible triplets
    triplets = list(product(side_one, side_two, side_three))
    
    return triplets
