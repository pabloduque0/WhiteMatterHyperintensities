import numpy as np
import cv2
import itk
import matplotlib.pyplot as plt
import constant

def leave_one_out(dataset, labels, dataset_name):

    test_image_idx = np.random(0, 19)
    test_indexes = None

    if dataset_name == 'utrecht':
        start = test_image_idx * constant.N_SLICE_UTRECHT
        end = test_image_idx * constant.N_SLICE_UTRECHT + constant.N_SLICE_UTRECHT
        test_image_idxes = np.arange(start, end)

    elif dataset_name == 'singapore':
        start = test_image_idx * constant.N_SLICE_SINGAPORE
        end = test_image_idx * constant.N_SLICE_SINGAPORE + constant.N_SLICE_SINGAPORE
        test_image_idxes = np.arange(start, end)

    elif dataset_name == 'amsterdam':
        start = test_image_idx * constant.N_SLICE_AMSTERDAM
        end = test_image_idx * constant.N_SLICE_AMSTERDAM + constant.N_SLICE_AMSTERDAM
        test_indexes = np.arange(start, end)
    else:
        print('Dataset name not found for LOO cross validation.')
        return None

    all_indexes = np.arange(0, len(dataset)-1)
    train_indexes = np.array(set(all_indexes).difference(set(test_indexes)))

    train_data, test_data = dataset[train_indexes], dataset[test_indexes]
    train_labels, test_labels = labels[train_indexes], labels[test_indexes]

    return train_data, test_data, train_labels, test_labels




