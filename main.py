import cv2
import numpy as np

def crossCorrelation():
    original = cv2.imread("test3.jpg")
    kernelOriginal = cv2.imread("sample2-2.jpg")

    base = cv2.imread("test3.jpg", cv2.IMREAD_GRAYSCALE)
    kernel = cv2.imread("sample2-2.jpg", cv2.IMREAD_GRAYSCALE)

    kernel = cv2.normalize(kernel, None, 0, 1, cv2.NORM_MINMAX)
    base = cv2.normalize(base, None, 0, 1, cv2.NORM_MINMAX)

    kernelHeight, kernelWidth = kernel.shape
    baseHeight, baseWidth = base.shape

    kernelNumpy = np.array(kernel)
    baseNumpy = np.array(base)

    maxV = 0

    result = np.zeros((baseHeight - kernelHeight,
                       baseWidth - kernelWidth))

    result_uint = np.zeros((baseHeight - kernelHeight,
                            baseWidth - kernelWidth), np.uint8)

    for y in range(0, baseHeight - kernelHeight):
        for x in range(0, baseWidth - kernelWidth):

            tot = 0
            tempBase = baseNumpy[y:y+kernelHeight, x:x+kernelWidth].copy()
            tempBase = tempBase*kernelNumpy

            tot = tempBase.sum()

            result[y, x] = tot
            result_uint[y, x] = tot

            if(maxV < tot):
                print(x, y)
                maxV = tot

    for i in range(0, baseHeight - kernelHeight):
        for j in range(0, baseWidth - kernelWidth):
            print(result[i, j], maxV * 0.99)
            if(result[i, j] >= maxV * 0.99):
                cv2.rectangle(original, (j, i), (j+kernelWidth,
                                                 i+kernelHeight), (255, 0, 0), 2)

    cv2.imshow('base', original)
    cv2.imshow('kernel', kernelOriginal)
    cv2.imshow('cross-correlation result', result_uint)
    cv2.waitKey(0)

crossCorrelation()
