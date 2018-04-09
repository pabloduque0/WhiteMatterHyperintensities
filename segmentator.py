import itk

class Segmentator():

    def __init__(self):
        pass


    def read_image(self, path):

        image = itk.imread(path)
        return image

    def normalize_image(self, image, alpha, beta, radius_value):
        histogramEqualization = itk.AdaptiveHistogramEqualizationImageFilter.New(image)
        histogramEqualization.SetAlpha(alpha)
        histogramEqualization.SetBeta(beta)
        radius = itk.Size[3]()
        radius.Fill(radius_value)

        histogramEqualization.SetRadius(radius)
        histogramEqualization.Update()
        return histogramEqualization.GetOutput()

    def threshold_image(self, image, upperthreshold, lowerthreshold, insidevalue, outsidevalue):

        pixeltype = itk.ctype(image.GetPixelIDTypeAsString())
        thresholdFilter = itk.BinaryThresholdImageFilter[pixeltype].New()
        thresholdFilter.SetInput(image)
        thresholdFilter.SetLowerThreshold(lowerthreshold)
        thresholdFilter.SetUpperThreshold(upperthreshold)
        thresholdFilter.SetOutsideValue(outsidevalue)
        thresholdFilter.SetInsideValue(insidevalue)