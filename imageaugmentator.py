#############################################################################
#                                                                           #
# BASED KERAS IMAGE AUGMENTATION:                                           #
# https://keras.io/preprocessing/image/                                     #
#                                                                           #
#############################################################################

import numpy as np
import scipy.ndimage as ndi


class ImageAugmentator():

    def __init__(self):
        pass

    def perform_all_augmentations(self, dataset_x, dataset_y):

        if len(dataset_x) != len(dataset_y):
            print('Wrong input :thumbs_down: . Image lists must be have the same length.')
            return

        all_indexes = np.arange(0, len(dataset_x)-1)
        half_indexes = np.random.randint(0, len(dataset_x)-1, len(dataset_x)//2)
        other_half_indexes = list(set(all_indexes).difference(set(half_indexes)))

        dataset_y_copy = np.concatenate([dataset_y, dataset_y, dataset_y], axis=3)

        # Rotations
        rotation_slice_x = dataset_x[half_indexes]
        rotation_slice_y = dataset_y_copy[half_indexes]
        rotated_xs, rotated_ys = self.perform_rotations(rotation_slice_x, rotation_slice_y, 10)
        aug_dataset_x = np.concatenate([dataset_x, rotated_xs], axis=0)
        rotated_ys = np.expand_dims(np.asanyarray(rotated_ys)[:, :, :, 0], axis=3)
        aug_dataset_y = np.concatenate([dataset_y, rotated_ys], axis=0)

        # Shifts
        shift_slice_x = dataset_x[other_half_indexes]
        shift_slice_y = dataset_y[other_half_indexes]
        shift_xs, shift_ys = self.perform_shifts(shift_slice_x, shift_slice_y, 0.3, 0.3)
        aug_dataset_x = np.concatenate([aug_dataset_x, shift_xs], axis=0)
        aug_dataset_y = np.concatenate([aug_dataset_y, shift_ys], axis=0)

        return aug_dataset_x, aug_dataset_y



    def perform_flips(self, images_x, images_y):

        if len(images_x) != len(images_y):
            print('Wrong input :thumbs_down: . Image lists must be have the same length.')
            return

        flipped_list_x = []
        flipped_list_y = []
        for image_x, image_y in zip(images_x, images_y):
            flipped_x, flipped_y = self.random_flip(image_x, image_y, 1)
            flipped_list_x.append(flipped_x)
            flipped_list_y.append(flipped_y)

        return flipped_list_x, flipped_list_y

    def perform_rotations(self, images_x, images_y, angle):

        if len(images_x) != len(images_y):
            print('Wrong input :thumbs_down: . Image lists must be have the same length.')
            return

        rotated_list_x = []
        rotated_list_y = []
        for image_x, image_y in zip(images_x, images_y):
            rotated_x, rotated_y = self.random_rotation(image_x, image_y, angle)
            rotated_list_x.append(rotated_x)
            rotated_list_y.append(rotated_y)

        return rotated_list_x, rotated_list_y


    def perform_shifts(self, images_x, images_y, width_shift, height_shift):

        if len(images_x) != len(images_y):
            print('Wrong input :thumbs_down: . Image lists must be have the same length.')
            return

        shifted_list_x = []
        shifted_list_y = []
        for image_x, image_y in zip(images_x, images_y):
            rotated_x, rotated_y = self.random_shift(image_x, image_y, width_shift, height_shift)
            shifted_list_x.append(rotated_x)
            shifted_list_y.append(rotated_y)

        return shifted_list_x, shifted_list_y

    def random_shift(self, x, y, wrg, hrg, row_axis=0, col_axis=1, channel_axis=2,
                     fill_mode='constant', cval=0.0):
        """Performs a random spatial shift of a Numpy image tensor.
        # Arguments
            x: Input tensor. Must be 3D.
            wrg: Width shift range, as a float fraction of the width.
            hrg: Height shift range, as a float fraction of the height.
            row_axis: Index of axis for rows in the input tensor.
            col_axis: Index of axis for columns in the input tensor.
            channel_axis: Index of axis for channels in the input tensor.
            fill_mode: Points outside the boundaries of the input
                are filled according to the given mode
                (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
            cval: Value used for points outside the boundaries
                of the input if `mode='constant'`.
        # Returns
            Shifted Numpy image tensor.
        """
        h, w = x.shape[row_axis], x.shape[col_axis]
        tx = np.random.uniform(-hrg, hrg) * h
        ty = np.random.uniform(-wrg, wrg) * w
        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])

        transform_matrix = translation_matrix  # no need to do offset
        x = self.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
        y = self.apply_transform(y, transform_matrix, channel_axis, fill_mode, cval)

        return x, y

    def random_flip(self, x, y, axis):

        x = np.asarray(x).swapaxes(axis, 0)
        x = x[::-1, ...]
        x = x.swapaxes(0, axis)

        y = np.asarray(y).swapaxes(axis, 0)
        y = y[::-1, ...]
        y = y.swapaxes(0, axis)

        return x, y

    def random_rotation(self, x, y, rg, row_axis=0, col_axis=1, channel_axis=2, fill_mode='constant', cval=0.0):
        """Performs a random rotation of a Numpy image tensor.
        # Arguments
            x: Input tensor. Must be 3D.
            rg: Rotation range, in degrees.
            row_axis: Index of axis for rows in the input tensor.
            col_axis: Index of axis for columns in the input tensor.
            channel_axis: Index of axis for channels in the input tensor.
            fill_mode: Points outside the boundaries of the input
                are filled according to the given mode
                (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
            cval: Value used for points outside the boundaries
                of the input if `mode='constant'`.
        # Returns
            Rotated Numpy image tensor.
        """

        if x.shape != y.shape:
            raise Exception('X and Y images must have same shape.')

        theta = np.deg2rad(np.random.uniform(-rg, rg))
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])


        h, w = x.shape[row_axis], x.shape[col_axis]
        transform_matrix = self.transform_matrix_offset_center(rotation_matrix, h, w)
        x = self.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
        y = self.apply_transform(y, transform_matrix, channel_axis, fill_mode, cval)

        return x, y
    
    def transform_matrix_offset_center(self, matrix, x, y):
        o_x = float(x) / 2 + 0.5
        o_y = float(y) / 2 + 0.5
        offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
        reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
        transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
        return transform_matrix


    def apply_transform(self, x, transform_matrix, channel_axis=0, fill_mode='constant', cval=0.0):
        """Apply the image transformation specified by a matrix.
        # Arguments
            x: 2D numpy array, single image.
            transform_matrix: Numpy array specifying the geometric transformation.
            channel_axis: Index of axis for channels in the input tensor.
            fill_mode: Points outside the boundaries of the input
                are filled according to the given mode
                (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
            cval: Value used for points outside the boundaries
                of the input if `mode='constant'`.
        # Returns
            The transformed version of the input.
        """
        x = np.rollaxis(x, channel_axis, 0)
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]
        channel_images = [ndi.interpolation.affine_transform(
            x_channel,
            final_affine_matrix,
            final_offset,
            order=1,
            mode=fill_mode,
            cval=cval) for x_channel in x]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_axis + 1)
        return x


