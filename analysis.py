import numpy as np
import matplotlib.pyplot as plt


def analyze_hit_intensities(t1s, flairs, labels):
    fig1, ax1 = plt.subplots(2, 1)
    all_flair_values = None
    all_t1_values = None
    for index, (t1, flair, label) in enumerate(zip(t1s, flairs, labels)):

        wmh_indexes = np.where(label == 1.0)
        wmh_values_t1 = t1[wmh_indexes]
        wmh_values_flair = flair[wmh_indexes]

        if len(wmh_indexes) > 0 and len(wmh_values_t1) > 0 and len(wmh_values_flair) > 0:

            if all_flair_values is None or all_t1_values is None:
                all_flair_values = wmh_values_flair
                all_t1_values = wmh_values_t1

            all_flair_values = np.concatenate([all_flair_values, wmh_values_flair])
            all_t1_values = np.concatenate([all_t1_values, wmh_values_t1])

    ax1[0].hist(all_flair_values, bins=40, color='r')
    ax1[0].set_title('Flair images')
    ax1[1].hist(all_t1_values, bins=40, color='g')
    ax1[1].set_title('T1 images')

    ax1[0].set_ylabel('Appeareances')

    ax1[1].set_xlabel('Pixel intensity')
    ax1[1].set_ylabel('Appeareances')

    plt.show()


def analyze_hits_locats(t1s, flairs, labels):
    all_indexes = None

    for index, (t1, flair, label) in enumerate(zip(t1s, flairs, labels)):
        wmh_indexes = np.where(label == 1.0)
        if len(wmh_indexes) > 0:
            if all_indexes is not None:
                all_indexes[0] = np.concatenate([all_indexes[0], wmh_indexes[0]])
                all_indexes[1] = np.concatenate([all_indexes[1], wmh_indexes[1]])
            else:
                all_indexes = list(wmh_indexes)

    plt.hist2d(all_indexes[0], all_indexes[1], bins=100, range=[[0, t1s[0].shape[0]], [0, t1s[0].shape[1]]],
               cmin=1)
    plt.colorbar()
    plt.show()