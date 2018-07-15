import os
import gzip
import shutil
import itk
import cv2
import numpy as np
import subprocess


class ImageParser():

    def __init__(self):
        self.path_utrech = '../Utrecht'
        self.path_singapore = '../Singapore'
        self.path_amsterdam = '../GE3T'


    def get_all_image_paths(self):
        paths = []

        for root, dirs, files in os.walk('../'):
            for file in files:
                filepath = root + '/' + file

                if file.endswith('.gz') and file[:-3] not in files:
                    with gzip.open(filepath, 'rb') as f_in:
                        with open(filepath[:-3], 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)

                if file.startswith('brain') and file.endswith('.nii'):
                    paths.append(filepath)

        return paths


    def get_all_images_and_labels(self):
        utrech_dataset = self.get_images_and_labels(self.path_utrech)
        singapore_dataset = self.get_images_and_labels(self.path_singapore)
        amsterdam_dataset = self.get_images_and_labels(self.path_amsterdam)

        return utrech_dataset, singapore_dataset, amsterdam_dataset


    def get_images_and_labels(self, path):
        full_dataset = []
        data_and_labels = []

        for root, dirs, files in os.walk(path):
            for file in files:
                filepath = root + '/' + file
                if file == 'wmh.nii':
                    data_and_labels.append(filepath)
                    print('here yes')

                if '/pre/' in filepath and (file == 'brain_FLAIR.nii' or file == 'brain_T1.nii') and len(
                        data_and_labels) in (1, 2):
                    data_and_labels.append(filepath)
                    if len(data_and_labels) == 3:
                        full_dataset.append(list(data_and_labels))
                        print(data_and_labels)
                        data_and_labels.clear()

        return full_dataset


    def get_all_images_itk(self, paths_list):
        images = []
        for path in paths_list:
            image = itk.imread(path)
            images.append(image)

        return images


    def get_all_images_np(self, paths_list, slice_shape, normalization=True):
        images = []
        for path in paths_list:
            image = itk.imread(path)
            np_image = itk.GetArrayFromImage(image)
            np_image = np.swapaxes(np_image, 0, 2)
            resized = self.threed_resize(np_image, slice_shape)
            np_image = np.swapaxes(resized, 0, 2)

            if normalization:
                normalized = self.normalize_image(np_image)
                if normalized is not None:
                    np_image = np.expand_dims(normalized, 4)
                    images.append(np_image)
            else:
                np_image[np_image > 1.] = 0.0
                np_image = np.expand_dims(np_image, 4)
                images.append(np_image)

        return images

    def get_all_images_np_twod(self, paths_list, slice_shape, normalization=True):

        images = []
        for path in paths_list:
            image = itk.imread(path)
            np_image = itk.GetArrayFromImage(image)
            np_image = np.swapaxes(np_image, 0, 2)
            resized = self.threed_resize(np_image, slice_shape)

            if normalization:
                normalized = self.normalize_image(resized)
                if normalized is not None:
                    np_image = np.expand_dims(normalized, 3)
                    images.append(np_image)
            else:
                resized[resized > 1.] = 0.0
                np_image = np.expand_dims(resized, 3)
                images.append(np_image)

        slices_list = self.get_slices_list(images)

        return slices_list

    def get_slices_list(self, images_list):

        slices = []
        for image in images_list:
            for slice in image:
                slices.append(slice)

        return np.asanyarray(slices)



    def threed_resize(self, image, slice_shape):

        all_slices = []
        for index in range(image.shape[2]):
            slice = image[:, :, index]
            resized = cv2.resize(slice, (slice_shape[1], slice_shape[0]))
            all_slices.append(resized)

        return np.asanyarray(all_slices)

    def display_image(self, image):
        np_image = itk.GetArrayFromImage(image)

        np_image = np.swapaxes(np_image, 0, 2)
        np_image = np_image.astype(np.uint8)
        print(np_image.dtype)
        rows, columns, slices = np_image.shape
        print(type(np_image))
        for slice in range(slices):
            slice_image = np_image[:, :, slice]
            cv2.imshow('Image', slice_image)
            cv2.waitKey(0)

    def normalize_image(self, image):
        non_black = image[image > 0]
        flattened_nonblack = np.ravel(non_black)
        sorted_data = sorted(flattened_nonblack)

        five_percent = int(len(sorted_data) * 0.05)
        lower_threshold = sorted_data[five_percent]
        upper_threshold = sorted_data[-five_percent]

        try:
            temp_matrix = np.maximum(0, (image - lower_threshold) / (upper_threshold - lower_threshold))
            normalized = np.minimum(1.0, temp_matrix)
            return normalized
        except Exception as e:
            print('Error in normalization: ', e)
            return None





    def extract_all_brains(self):
        base_command = 'fsl5.0-bet '
        brain_str = 'brain_'
        for root, dirs, files in os.walk('../'):
            for file in files:
                filepath = root + '/' + file

                if '.nii' in file and file != 'wmh.nii' and 'mask' not in file:
                    full_command = base_command + filepath + ' ' + root + '/' + brain_str + file
                    process = subprocess.Popen(full_command.split(), stdout=subprocess.PIPE)
                    output, error = process.communicate()

                    print('OUTPUT: ', output)
                    print('ERROR: ', error)