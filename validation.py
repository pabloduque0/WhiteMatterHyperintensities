#############################################################################
#                                                                           #
# BASED ON WMH CHALLENGE EVALUATION FILE:                                   #
# https://github.com/hjkuijf/wmhchallenge/blob/master/evaluation.py         #
#                                                                           #
#############################################################################

import numpy as np
from twodunet import TwoDUnet
from imageparser import ImageParser
from imageaugmentator import ImageAugmentator
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model


def getDSC(labels, predictions):
    """Compute the Dice Similarity Coefficient."""
    return (1.0 - scipy.spatial.distance.dice(labels, predictions))


def getHausdorff(labels, predictions):
    """Compute the Hausdorff distance."""


    # Hausdorff distance is only defined when something is detected
    resultStatistics = sitk.StatisticsImageFilter()
    resultStatistics.Execute(predictions)
    if resultStatistics.GetSum() == 0:
        return float('nan')

    # Edge detection is done by ORIGINAL - ERODED, keeping the outer boundaries of lesions. Erosion is performed in 2D
    eTestImage = sitk.BinaryErode(labels, (1, 1, 0))
    eResultImage = sitk.BinaryErode(predictions, (1, 1, 0))

    hTestImage = sitk.Subtract(labels, eTestImage)
    hResultImage = sitk.Subtract(predictions, eResultImage)

    hTestArray = sitk.GetArrayFromImage(hTestImage)
    hResultArray = sitk.GetArrayFromImage(hResultImage)

    # Convert voxel location to world coordinates. Use the coordinate system of the test image
    # np.nonzero   = elements of the boundary in numpy order (zyx)
    # np.flipud    = elements in xyz order
    # np.transpose = create tuples (x,y,z)
    # labels.TransformIndexToPhysicalPoint converts (xyz) to world coordinates (in mm)
    testCoordinates = np.apply_along_axis(labels.TransformIndexToPhysicalPoint, 1,
                                          np.transpose(np.flipud(np.nonzero(hTestArray))).astype(int))
    resultCoordinates = np.apply_along_axis(labels.TransformIndexToPhysicalPoint, 1,
                                            np.transpose(np.flipud(np.nonzero(hResultArray))).astype(int))

    # Compute distances from test to result; and result to test
    dTestToResult = getDistancesFromAtoB(testCoordinates, resultCoordinates)
    dResultToTest = getDistancesFromAtoB(resultCoordinates, testCoordinates)

    return max(np.percentile(dTestToResult, 95), np.percentile(dResultToTest, 95))

# Use a kd-tree for fast spatial search
def getDistancesFromAtoB(a, b):
    kdTree = scipy.spatial.KDTree(a, leafsize=100)
    return kdTree.query(b, k=1, eps=0, p=2)[0]

def getLesionDetection(labels, predictions):
    """Lesion detection metrics, both recall and F1."""

    # Connected components will give the background label 0, so subtract 1 from all results
    ccFilter = sitk.ConnectedComponentImageFilter()
    ccFilter.SetFullyConnected(True)

    # Connected components on the test image, to determine the number of true WMH.
    # And to get the overlap between detected voxels and true WMH
    ccTest = ccFilter.Execute(labels)
    lResult = sitk.Multiply(ccTest, sitk.Cast(predictions, sitk.sitkUInt32))

    ccTestArray = sitk.GetArrayFromImage(ccTest)
    lResultArray = sitk.GetArrayFromImage(lResult)

    # recall = (number of detected WMH) / (number of true WMH)
    nWMH = len(np.unique(ccTestArray)) - 1
    if nWMH == 0:
        recall = 1.0
    else:
        recall = float(len(np.unique(lResultArray)) - 1) / nWMH

    # Connected components of results, to determine number of detected lesions
    ccResult = ccFilter.Execute(predictions)
    lTest = sitk.Multiply(ccResult, sitk.Cast(labels, sitk.sitkUInt32))

    ccResultArray = sitk.GetArrayFromImage(ccResult)
    lTestArray = sitk.GetArrayFromImage(lTest)

    # precision = (number of detections that intersect with WMH) / (number of all detections)
    nDetections = len(np.unique(ccResultArray)) - 1
    if nDetections == 0:
        precision = 1.0
    else:
        precision = float(len(np.unique(lTestArray)) - 1) / nDetections

    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * (precision * recall) / (precision + recall)

    return recall, f1


def getAVD(labels, predictions):
    """Volume statistics."""
    # Compute statistics of both images
    testStatistics = sitk.StatisticsImageFilter()
    resultStatistics = sitk.StatisticsImageFilter()

    testStatistics.Execute(labels)
    resultStatistics.Execute(predictions)

    return float(abs(testStatistics.GetSum() - resultStatistics.GetSum())) / float(testStatistics.GetSum()) * 100