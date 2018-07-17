import numpy as np
from twodunet import TwoDUnet
from imageparser import ImageParser


parser = ImageParser()

utrech_dataset, singapore_dataset, amsterdam_dataset = parser.get_all_images_and_labels()

t1_utrecht = [row[1] for row in utrech_dataset]
flair_utrecht = [row[2] for row in utrech_dataset]
labels_utrecht = [row[0] for row in utrech_dataset]

t1_singapore = [row[1] for row in singapore_dataset]
flair_singapore = [row[1] for row in singapore_dataset]
labels_singapore = [row[0] for row in singapore_dataset]

#data_amsterdam = [row[1] for row in amsterdam_dataset]
#labels_amsterdam = [row[0] for row in amsterdam_dataset]

all_t1 = t1_utrecht + t1_singapore # + list(data_amsterdam)
all_flair = flair_utrecht + flair_singapore

all_labels = labels_utrecht + labels_singapore # + labels_amsterdam


slice_shape = (240, 240)

data_t1 = parser.get_all_images_np_twod(all_t1, slice_shape)
data_flair = parser.get_all_images_np_twod(all_flair, slice_shape)
print(data_t1[0].shape)
print(data_flair[0].shape)
all_data = np.concatenate([data_t1, data_flair], axis=3)
labels = parser.get_all_images_np_twod(all_labels, slice_shape, normalization=False)

data = np.asanyarray(all_data)
labels = np.asanyarray(labels)

print('Maxes: ', np.max(data), np.max(labels))
print('Mins: ', np.min(data), np.min(labels))


print(data.shape, labels.shape)

unet = TwoDUnet(model_path=None, img_shape=data.shape[1:])

training_name = 'third_test'
base_path = '/home/pablo/Google Drive/UNED/Vision_Artificial/M2/WhiteMatterHyperintensities/'
test_size = 0.3

print(data.shape, labels.shape)

unet.train(data, labels, test_size, training_name, base_path, epochs=10, batch_size=1)
