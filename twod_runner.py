import numpy as np
from twodunet import TwoDUnet
from imageparser import ImageParser
from imageaugmentator import ImageAugmentator

parser = ImageParser()
utrech_dataset, singapore_dataset, amsterdam_dataset = parser.get_all_images_and_labels()

t1_utrecht = [row[1] for row in utrech_dataset]
flair_utrecht = [row[2] for row in utrech_dataset]
labels_utrecht = [row[0] for row in utrech_dataset]

t1_singapore = [row[1] for row in singapore_dataset]
flair_singapore = [row[2] for row in singapore_dataset]
labels_singapore = [row[0] for row in singapore_dataset]

#data_amsterdam = [row[1] for row in amsterdam_dataset]
#labels_amsterdam = [row[0] for row in amsterdam_dataset]

all_t1_paths = t1_utrecht + t1_singapore # + list(data_amsterdam)
all_flair_paths = flair_utrecht + flair_singapore

slice_shape = (240, 240)

# All original data as np
data_t1 = parser.get_all_images_np_twod(all_t1_paths, slice_shape)
data_flair = parser.get_all_images_np_twod(all_flair_paths, slice_shape)
data_tophat = parser.generate_tophat(data_flair)
all_data = np.concatenate([data_t1, data_flair, data_tophat], axis=3)

# All labels as np
all_labels_paths = labels_utrecht + labels_singapore
labels_images = parser.get_all_images_np_twod(all_labels_paths, slice_shape, normalization=False)

# Augmented data as np
augmentator = ImageAugmentator()
data_augmented, labels_agumented = augmentator.perform_all_augmentations(all_data, labels_images)

data_augmented = np.asanyarray(data_augmented)
labels_agumented = np.asanyarray(labels_agumented)

print('Maxes: ', np.max(data_augmented), np.max(labels_images))
print('Mins: ', np.min(data_augmented), np.min(labels_images))

print(data_augmented.shape, labels_agumented.shape)

unet = TwoDUnet(model_path=None, img_shape=data_augmented.shape[1:])

training_name = 'data_augmentation_test'
base_path = '/harddrive/home/pablo/Google Drive/UNED/Vision_Artificial/M2/WhiteMatterHyperintensities'
test_size = 0.3

print(data_augmented.shape, labels_agumented.shape)

unet.train(data_augmented, labels_agumented, test_size, training_name, base_path, epochs=10, batch_size=1)
