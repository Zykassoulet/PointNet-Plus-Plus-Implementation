import zipfile
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt

data_path = '../Datasets/ShapeNetPart2019/bad/sem_seg_h5/'

obj = 'Knife-3/'
# Read H5 file
f = h5.File(data_path+obj+"test-00.h5", "r")

print(f.keys())
# Get and print list of datasets within the H5 file
data = f['data']  # points position
data_num = f['data_num']  # number of points
label_seg = f['label_seg']  # label of points


def all_label(label_list):
    labels = []
    for label in label_list:
        if not label in labels:
            labels.append(label)
    return labels


def load_label(fn):
    with open(fn, 'r') as fin:
        label = np.array([int(item.rstrip())
                         for item in fin.readlines()], dtype=np.int32)
        return label


print(all_label(label_seg[0]))
print(data.shape)
print(data[0])
