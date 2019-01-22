# -*- coding: utf-8 -*-

import os
import sys
import cv2
import h5py
import numpy as np
import pandas as pd
import random


def get_marked_img_names(info_filename):
    df = pd.read_table(info_filename, index_col=0, delim_whitespace=True)

    all = set(df.index)

    wearing_glasses = set(df[df['wearing_glasses'] == 1].index)
    wearing_sunglasses = set(df[df['wearing_sunglasses'] == 1].index)
    glasses = wearing_glasses | wearing_sunglasses

    no_glasses = random.sample(all - glasses, len(glasses))
    glasses = list(glasses)

    no_glasses_marked = [[x, 0] for x in no_glasses]
    glasses_marked = [[x, 1] for x in glasses]

    size = len(no_glasses_marked)
    indices = [int(size * 0.6), int(size * 0.8)]

    train = glasses_marked[:indices[0]] + no_glasses_marked[:indices[0]]
    random.shuffle(train)

    val = glasses_marked[indices[0]:indices[1]] + no_glasses_marked[indices[0]:indices[1]]
    random.shuffle(val)

    test = glasses_marked[indices[1]:] + no_glasses_marked[indices[1]:]
    random.shuffle(test)

    return train, test, val


def create_hdf5(dir_path, img_names, hdf5_path, block_size=128):
    print('Create "{}"...'.format(hdf5_path))
    img_count = len(img_names)

    f = h5py.File(hdf5_path, 'w')
    dset_data = f.create_dataset('data',
                                 shape=(img_count, block_size, block_size, 1),
                                 dtype=np.float32,
                                 maxshape=(None, block_size, block_size, 1),
                                 chunks=True)
    dset_label = f.create_dataset('label',
                                  shape=(img_count, 1),
                                  dtype=np.float32,
                                  maxshape=(None, 1),
                                  chunks=True)

    dset_idx = 0
    for img_path, label in ([os.path.join(dir_path, name[0] + '.jpg'), name[1]] for name in img_names):
        sys.stdout.write('\r{}/{}...'.format(dset_idx, img_count))
        sys.stdout.flush()

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (block_size, block_size))

        data = img / 255.
        data.resize((block_size, block_size, 1))

        dset_data[dset_idx] = data
        dset_label[dset_idx] = label
        f.flush()
        dset_idx += 1

    f.close()
    print('Создана база: ' + hdf5_path + '\n')


def read_hdf5(filename):
    with h5py.File(filename, 'r') as f:
        data = f['data'][:]
        label = f['label'][:]
        print(type(data), data.dtype, data.shape)
        print(type(label), label.dtype, label.shape)
        return data, label


if __name__ == "__main__":

    train, test, val = get_marked_img_names('dataset/Selfie-dataset/selfie_dataset.txt')

    dir_path = 'dataset/Selfie-dataset/images'
    create_hdf5(dir_path, train, 'dataset/train.h5')
    create_hdf5(dir_path, test, 'dataset/test.h5')
    create_hdf5(dir_path, val, 'dataset/val.h5')

    read_hdf5('dataset/train.h5')
    read_hdf5('dataset/test.h5')
    read_hdf5('dataset/val.h5')
