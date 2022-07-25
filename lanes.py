import cv2
from random import *
import numpy as np


def ResizeRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)

def show(title, image):
    cv2.imshow(title, image)
    cv2.waitKey()

def vid_show(title, stream):
    cv2.imshow(title, stream)
    key = cv2.waitKey(1)
    return key

def gray(image):
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_frame

def edge(image):
    sigma = 0.33
    median = np.median(image)
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    edge_image = cv2.Canny(image, lower, upper)
    return edge_image

def threshold(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    seg_image = cv2.inRange(hsv, np.array(
        [-10, 50, 100]), np.array([50, 150, 225]))
    return seg_image

def getROI(image):
    height = image.shape[0]
    width = image.shape[1]
    # Defining Triangular ROI: The values will change as per your camera mounts
    triangle = np.array(
        [[(100, height), (width, height), (width-500, int(height/1.9))]])
    # creating black image same as that of input image
    black_image = np.zeros_like(image)
    # Put the Triangular shape on top of our Black image to create a mask
    mask = cv2.fillPoly(black_image, triangle, 255)
    # applying mask on original image
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def getLines(image):
    lines = cv2.HoughLinesP(image, 0.3, np.pi/180, 100,
                            np.array([]), minLineLength=70, maxLineGap=20)
    return lines

def displayLines(image, lines):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)  # converting to 1d array
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return image

def getLineCoordinatesFromParameters(image, line_parameters):
    slope = line_parameters[0]
    intercept = line_parameters[1]
    y1 = image.shape[0]  # since line will always start from bottom of image
    y2 = int(y1 * (3.4 / 5))  # some random point at 3/5
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def getSmoothLines(image, lines):
    left_fit = []  # will hold m,c parameters for left side lines
    right_fit = []  # will hold m,c parameters for right side lines

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]

        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)

    # now we have got m,c parameters for left and right line, we need to know x1,y1 x2,y2 parameters
    left_line = getLineCoordinatesFromParameters(image, left_fit_average)
    right_line = getLineCoordinatesFromParameters(image, right_fit_average)
    return np.array([left_line, right_line])


# image = cv2.imread("D:\Projects\Python\CyberTruck\sample.png")
# cv2.imshow("Output", image_with_smooth_lines)
# cv2.waitKey(0)

cam_stream = cv2.VideoCapture(0)

while True:
    is_frame, frame = cam_stream.read()
    frame = ResizeRatio(frame, height=600)
    edged_stream = edge(frame)
    roi_stream = getROI(edged_stream)
    lines = getLines(roi_stream)
    # smooth_lines = getSmoothLines(frame, lines)
    # stream_with_smooth_lines = displayLines(frame, smooth_lines)
    key = vid_show("Detect x Lane", roi_stream)
    if key == ord('q'):
        break


"""
cam_stream = cv2.VideoCapture(0)
while True:
    is_frame, frame = cam_stream.read()
    frame = ResizeRatio(frame, height=600)
    gray_frame = gray(frame)
    edge_image = edge(frame)
    seg_image = threshold(frame)
    key = vid_show("Detect x Lane", seg_image)
    if key == ord('q'):
        break

cam_stream.release()
"""

cv2.destroyAllWindows()
