import numpy as np
import cv2
import itk
import matplotlib.pyplot as plt

class Utils():

    def calc_3d_hist(self, itk_image):

        slices_array = self.to_slices_array(itk_image)
        hist = cv2.calcHist(slices_array[20], [0], None, [256], [0, 256])
        plt.hist(slices_array[20].ravel(), 256, [0, 256])
        plt.show()


    def display_image(self, image):

        np_image = self.to_numpy_corrected(image)
        rows, columns, slices = np_image.shape
        print(type(np_image))
        for slice in range(slices):
            slice_image = np_image[:, :, slice]
            cv2.imshow('Image', slice_image)
            cv2.waitKey(0)

    def to_numpy_corrected(self, image):

        np_image = itk.GetArrayFromImage(image)
        np_image = np.swapaxes(np_image, 0, 2)
        # = np_image.astype(np.uint8)

        return np_image

    def to_slices_array(self, image):

        np_image = self.to_numpy_corrected(image)

        slices_array = [np_image[:, :, _slice] for _slice in range(np_image.shape[2])]
        return slices_array