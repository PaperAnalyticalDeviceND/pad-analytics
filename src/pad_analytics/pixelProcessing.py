import cv2 as cv
import numpy as np


# Takes a list of pixels and a BGR image and returns the average
# RGB pixel values
def avgPixels(pixels, img):
    totalB = 0.0  # Use float to prevent overflow
    totalG = 0.0
    totalR = 0.0
    for pixel in pixels:
        x = pixel[0]
        y = pixel[1]
        b, g, r = img[x, y, :]
        # Convert to float to prevent overflow
        totalB += float(b)
        totalG += float(g)
        totalR += float(r)
    if len(pixels) != 0:
        totalB /= len(pixels)
        totalG /= len(pixels)
        totalR /= len(pixels)
    return int(totalR + 0.5), int(totalG + 0.5), int(totalB + 0.5)


# Takes a list of pixels and a BGR image and returns the average
# HSV pixel values
def avgPixelsHSV(pixels, img):
    workingImg = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    totalH = 0.0  # Use float to prevent overflow
    totalS = 0.0
    totalV = 0.0
    for pixel in pixels:
        x = pixel[0]
        y = pixel[1]
        h, s, v = workingImg[x, y, :]
        # Convert to float to prevent overflow
        totalH += float(h)
        totalS += float(s)
        totalV += float(v)
    if len(pixels) != 0:
        totalH /= len(pixels)
        totalS /= len(pixels)
        totalV /= len(pixels)
    return totalH, totalS, totalV


# Takes a list of pixels and a BGR image and returns the average
# Lab pixel values
def avgPixelsLAB(pixels, img):
    workingImg = cv.cvtColor(img, cv.COLOR_BGR2Lab)
    totalL = 0.0  # Use float to prevent overflow
    totalA = 0.0
    totalB = 0.0
    for pixel in pixels:
        x = pixel[0]
        y = pixel[1]
        l, a, b = workingImg[x, y, :]
        # Convert to float to prevent overflow
        totalL += float(l)
        totalA += float(a)
        totalB += float(b)
    if len(pixels) != 0:
        totalL /= len(pixels)
        totalA /= len(pixels)
        totalB /= len(pixels)
    return int(totalL + 0.5), int(totalA + 0.5), int(totalB + 0.5)
