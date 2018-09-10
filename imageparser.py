import os
import gzip
import shutil
import itk
import cv2
import numpy as np
import subprocess

class ImageParser():

    def __init__(self, path_utrech='../Utrecht', path_singapore='../Singapore', path_amsterdam='../GE3T'):
        self.path_utrech = path_utrech
        self.path_singapore = path_singapore
        self.path_amsterdam = path_amsterdam


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

    def get_all_images_np_twod(self, paths_list):

        slices_list = []
        for path in paths_list:
            image = itk.imread(path)
            np_image = itk.GetArrayFromImage(image)
            if np_image.shape[1:] == (232, 256):
                np_image = np.swapaxes(np_image, 1, 2)
                print('Corrected axises')

            for slice in np_image:
                slices_list.append(slice)

        return slices_list

    def get_slices_list(self, images_list):

        slices = []
        for image in images_list:
            for slice in image:
                slices.append(slice)

        return np.asanyarray(slices)

    def resize_slices(self, slices_list, to_slice_shape):

        resized_slices = []

        for slice in slices_list:
            slice_copy = slice.copy()

            if slice.shape[0] < to_slice_shape[0]:
                diff = to_slice_shape[0] - slice.shape[0]
                if self.is_odd(diff):
                    slice_copy = cv2.copyMakeBorder(slice_copy, diff//2, diff//2 + 1, 0, 0,
                                                    cv2.BORDER_CONSTANT,
                                                    value=0.0)
                else:
                    slice_copy = cv2.copyMakeBorder(slice_copy, diff // 2, diff // 2, 0, 0,
                                                    cv2.BORDER_CONSTANT,
                                                    value=0.0)

            elif slice.shape[0] > to_slice_shape[0]:
                diff = slice.shape[0] - to_slice_shape[0]
                if self.is_odd(diff):
                    slice_copy = slice_copy[diff//2 : -diff//2 + 1, :]
                else:
                    slice_copy = slice_copy[diff // 2: -diff // 2, :]

            if slice.shape[1] < to_slice_shape[1]:
                diff = to_slice_shape[1] - slice.shape[1]
                if self.is_odd(diff):
                    slice_copy = cv2.copyMakeBorder(slice_copy, 0, 0, diff // 2, diff // 2 + 1,
                                                    cv2.BORDER_CONSTANT,
                                                    value=0.0)
                else:
                    slice_copy = cv2.copyMakeBorder(slice_copy, 0, 0, diff // 2, diff // 2,
                                                    cv2.BORDER_CONSTANT,
                                                    value=0.0)
            elif slice.shape[1] > to_slice_shape[1]:
                diff = slice.shape[1] - to_slice_shape[1]
                if self.is_odd(diff):
                    slice_copy = slice_copy[:, diff // 2: -diff // 2 + 1]
                else:
                    slice_copy = slice_copy[:, diff // 2: -diff // 2]

            resized_slices.append(slice_copy)

        return resized_slices

    def is_odd(self, number):

        return number % 2 != 0

    def threed_resize(self, image, slice_shape):

        all_slices = []
        for index in range(image.shape[2]):
            slice = image[:, :, index]
            resized = cv2.resize(slice, (slice_shape[1], slice_shape[0]), cv2.INTER_CUBIC)
            all_slices.append(resized)

        return np.asanyarray(all_slices)

    def display_image(self, image):
        np_image = itk.GetArrayFromImage(image)

        np_image = np.swapaxes(np_image, 0, 2)
        np_image = np_image.astype(np.uint8)
        rows, columns, slices = np_image.shape
        for slice in range(slices):
            slice_image = np_image[:, :, slice]
            cv2.imshow('Image', slice_image)
            cv2.waitKey(0)

    def normalize_images(self, images_list):

        normalized_list = []

        np_list = np.concatenate(images_list, axis=1)
        flattened = np.ravel(np_list)
        non_black = flattened[flattened > 0]
        flattened_nonblack = np.ravel(non_black)
        sorted_data = sorted(flattened_nonblack)

        five_percent = int(len(sorted_data) * 0.05)
        lower_threshold = sorted_data[five_percent]
        upper_threshold = sorted_data[-five_percent]
        full_max = np.max(flattened)

        for slice in images_list:

            upper_indexes = np.where(slice >= upper_threshold)
            lower_indexes = np.where(slice <= lower_threshold)

            slice[upper_indexes] = 1.0
            slice[lower_indexes] = 0.0

            normalized = slice / full_max

            normalized_list.append(normalized)

        return normalized_list

    def remove_third_label(self, labels_list):

        new_labels_list = []

        for image in labels_list:
            image[image > 1.] = 0.0
            new_labels_list.append(image)

        return new_labels_list



    def generate_tophat(self, dataset):

        tophat_list = []
        kernel = np.ones((3, 3))
        for image in dataset:
            tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
            tophat_list.append(np.expand_dims(tophat, axis=2))

        return tophat_list


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