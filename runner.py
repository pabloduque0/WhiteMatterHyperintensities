from threedunet import ThreeDUnet
from imageparser import ImageParser
import numpy as np


parser = ImageParser()
'''
image_paths = parser.get_all_image_paths()

parser.get_all_images(image_paths)
'''

utrech_dataset, singapore_dataset, amsterdam_dataset = parser.get_all_images_and_labels()

data = [row[2] for row in utrech_dataset]
labels = [row[0] for row in utrech_dataset]

'''
for paths in utrech_dataset:
    images = []
    for path in paths:
        image = segmentator.read_image('../Singapore/50/pre/brain_FLAIR.nii')
        images.append(image)

    data.append(images[0]) # Only one MRI image for now
    data.append(images[1]) # Only one MRI image for now
'''


data = parser.get_all_images_np(data)
labels = parser.get_all_images_np(labels)
data = np.asanyarray(data)
labels = np.asanyarray(labels)

unet = ThreeDUnet(model_path=None, img_shape=data.shape[1:])

training_name = 'first_test'
base_path = '/home/pablo/Google Drive/UNED/Vision_Artificial/M2/WhiteMatterHyperintensities/'
test_size=0.3

print(data.shape, labels.shape)

unet.train(data, labels, test_size, training_name, base_path, epochs=10, batch_size=1)

'''
utils = Utils()


unet = Unet()
trainer = Trainer(unet, optimizer='adam')
path = trainer.train(image,
                     'unet_test',
                     training_iters=32,
                     epochs=1,
                     dropout=0.5,
                     display_step=1)

'''
#normalized = segmentator.normalize_image(image)

#hist = segmentator.normalize_image(image, 0.5, 0.2, 5)
#itk.imwrite(normalized, 'normalized_test.nii')

