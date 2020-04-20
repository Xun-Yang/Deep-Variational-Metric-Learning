# -*- coding: utf-8 -*-


import collections
import numpy as np
from sklearn.preprocessing import LabelEncoder
from fuel.streams import DataStream
from fuel.schemes import IterationScheme, BatchSizeScheme, SequentialScheme, ShuffledScheme

from .cars196_dataset import Cars196Dataset
from .random_fixed_size_crop_mod import RandomFixedSizeCrop

import random

def get_streams(batch_size=50, dataset='cars196',crop_size=224, load_in_memory=False):
    '''
    args:
        batch_size (int):
            number of examples per batch
        dataset (str):
            specify the dataset from 'cars196', 'cub200_2011', 'products'.
        method (str or fuel.schemes.IterationScheme):
            batch construction method. Specify 'n_pairs_mc', 'clustering', or
            a subclass of IterationScheme that has constructor such as
            `__init__(self, batch_size, dataset_train)` .
        crop_size (int or tuple of ints):
            height and width of the cropped image.
    '''

    if dataset == 'cars196':
        dataset_class = Cars196Dataset
    else:
        raise ValueError(
            "`dataset` must be 'cars196', 'cub200_2011' or 'products'.")

    dataset_train = dataset_class(['train'], load_in_memory=load_in_memory)
    dataset_test = dataset_class(['test'], load_in_memory=load_in_memory)

    if not isinstance(crop_size, tuple):
        crop_size = (crop_size, crop_size)

    stream = DataStream(dataset_train, iteration_scheme=ShuffledScheme(dataset_train.num_examples, batch_size))
    stream_train = RandomFixedSizeCrop(stream, which_sources=('images',),random_lr_flip=True,window_shape=crop_size)
    stream_train_eval = RandomFixedSizeCrop(DataStream(dataset_train, iteration_scheme=SequentialScheme(dataset_train.num_examples, batch_size)),which_sources=('images',), center_crop=True, window_shape=crop_size)
    stream_test = RandomFixedSizeCrop(DataStream(dataset_test, iteration_scheme=SequentialScheme(dataset_test.num_examples, batch_size)),which_sources=('images',), center_crop=True, window_shape=crop_size)

    return stream_train, stream_train_eval, stream_test