from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import os
import collections

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

# find the cnts in the edge map, then initialize the contour that corresponds to the document
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
"""
RETR_EXTERNAL:
If you use this flag, it returns only extreme outer flags. All child cnts are left behind.
We can say, under this law, Only the eldest in every family is taken care of.
It doesn't care about other members of the family :).

CHAIN_APPROX_SIMPLE:
we need just two end points of that line. This is what cv.CHAIN_APPROX_SIMPLE does.
It removes all redundant points and compresses the contour, thereby saving memory.

"""

cnts = imutils.grab_contours(cnts)
docCnt = None

# ensure that at least one contour was found
if len(cnts) > 0:
    # sort the cnts according to their size in descending order
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # loop over the sorted contour
    for contour in cnts:
        # aproximate the contour
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # if our approximated contour has four points, then we can assume we have found the paper
        if len(approximation) == 4:
            docCnt = approximation
            break

# apply a four point perspective transform to both the
# original image and grayscale image to obtain a top-down
# birds eye view of the paper
paper = four_point_transform(image, docCnt.reshape(4, 2))
warped = four_point_transform(gray, docCnt.reshape(4, 2))

# apply Otsu's thresholding method to binarize the warped piece of paper
thresh = cv2.threshold(warped, thresh=0, maxval=255, type=cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

cv2.imshow("thresh", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
# find cnts in the thresholded image, then initialize
# the list of cnts that correspond to questions
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
questioncnts = []

# loop over cnts
for contour in cnts:
    # compute the bounding box of the contour, then use the bounding box to derive aspect ratio
    (x, y, w, h) = cv2.boundingRect(contour)
    ar = w / float(h)

    # in order to label the contour as a question, region should be sufficiently wide,
    # sufficiently tall, and have an aspect ratio approximately equal to 1
    if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
        questioncnts.append(contour)

color = (0, 0, 255)  # red
question = cv2.drawContours(paper, questioncnts, -1, color, 3)
cv2.imshow("Question", question)
cv2.waitKey(0)
cv2.destroyAllWindows()

# sort the question cnts top-to-bottom, then initiaize the total number of correct answers
questioncnts = contours.sort_contours(questioncnts, method="top-to-bottom")[0]
correct = 0

Color = collections.namedtuple("Color", ["blue", "green", "red"])
colors = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (255, 255, 0),
    "purple": (225, 0, 225),
}

# each question has 5 possible answers, to loop over the question in batches of 5
for (q, i) in enumerate(np.arange(0, len(questioncnts), 5)):
    # sort the cnts for the current question from left to right, then initialize the index
    # of the bubbled answer
    cnts = contours.sort_contours(questioncnts[i : i + 5])[0]
    rows_of_bubbles = cv2.drawContours(paper, cnts, -1, list(colors.values())[q], 3)
    cv2.imshow("rows bubble", rows_of_bubbles)
    cv2.waitKey(0)
    bubbled = None

    # loop over the sorted contours
    for (j, c) in enumerate(cnts):
        # construct a mask to the threshold image, then count the number of non-zero
        # pixels in the bubble area
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)

        # apply the mask to the thresholded image, then
        # count the number of non-zero pixels in the
        # bubble area
        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        total = cv2.countNonZero(mask)

        # if the current total has a larger number of total non-zero pixels, then we are
        # examining the currently bubbled-in answer
        if bubbled is None or total > bubbled[0]:
            bubbled = (total, j)

        bubble_hearthstone = cv2.drawContours(mask, [c], -1, 255, 3)
        cv2.imshow("bubbled", bubble_hearthstone)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
# python test_grader.py --image images/test_01.png
