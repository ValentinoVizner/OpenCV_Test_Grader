from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import os

os.putenv("DISPLAY", ":1.0")

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    "-i",
    "--image",
    required=True,
    help="/home/vaba/Desktop/podatci/Python/Python_PyImageSearch_DL4Cv/Day_4_test_grader/test_grader_OpenCV/",
)
args = vars(ap.parse_args())

# define the answer key which maps the question number
# to the correct answer
ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}

# load the image, convert it to grayscale, blur it slightly, then find the edges
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)

cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# python test_grader.py --image images/test_01.png
