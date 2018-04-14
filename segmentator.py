import itk
from utils import Utils
import numpy as np


class Segmentator():

    def __init__(self):
        self.utils = Utils()


    def read_image(self, path):

        image = itk.imread(path)
        return image

    def normalize_image(self, itk_image):
        np_image = self.utils.to_numpy_corrected(itk_image)
        non_black = np_image[np_image > 0]
        flattened_nonblack = np.ravel(non_black)
        sorted_data = sorted(flattened_nonblack)

        five_percent = int(len(sorted_data)*0.05)
        lower_threshold = sorted_data[five_percent]
        upper_threshold = sorted_data[-five_percent]

        flattened = np.ravel(np_image)
        for index0, row in enumerate(np_image):
            for index1, column in enumerate(row):
                for index2, cell in enumerate(column):
                    np_image[index0, index1, index2] = max(0, (cell - lower_threshold) / (upper_threshold - lower_threshold))


        itk_np_copy = itk.GetImageFromArray(np_image)
        return itk_np_copy


    def normalize_image_with_itk(self, itk_image, alpha, beta, radius_value):
        histogramEqualization = itk.AdaptiveHistogramEqualizationImageFilter.New(itk_image)
        histogramEqualization.SetAlpha(alpha)
        histogramEqualization.SetBeta(beta)
        radius = itk.Size[3]()
        radius.Fill(radius_value)

        histogramEqualization.SetRadius(radius)
        histogramEqualization.Update()
        return histogramEqualization.GetOutput()

    def threshold_image(self, itk_image, upperthreshold, lowerthreshold, insidevalue, outsidevalue):

        pixeltype = itk.ctype(itk_image.GetPixelIDTypeAsString())
        thresholdFilter = itk.BinaryThresholdImageFilter[pixeltype].New()
        thresholdFilter.SetInput(itk_image)
        thresholdFilter.SetLowerThreshold(lowerthreshold)
        thresholdFilter.SetUpperThreshold(upperthreshold)
        thresholdFilter.SetOutsideValue(outsidevalue)
        thresholdFilter.SetInsideValue(insidevalue)