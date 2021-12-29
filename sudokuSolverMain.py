import cv2
import numpy as np
from tensorflow.keras.models import load_model
from usefulFunctions import *
import sys

orig_stdout = sys.stdout
f=open('computationalDetails.txt','w')
sys.stdout = f
print('Computational data generated during calculation of the sudoku in the scanned image appears here')

# IMAGE VARIABLES AT ONE PLACE
pathImage = "sudokuResource.jpeg"
heightImage = 450
widthImage = 450
model = load_model('model.h5')

# PREPARING THE IMAGE TO MATCH TYPE AND STANDARDS OF MNIST DATA SET WHICH WAS USED FOR TRAINING NEURAL NETWORK
img = cv2.imread(pathImage)
img = cv2.resize(img, (widthImage, heightImage)) #making image to be a square
imgBlank = np.zeros((heightImage,widthImage,3), np.uint8) #creating blank image for any future emergencies
imgThreshold = preProcess(img)

# FINDING ALL CONTOURS IN THE IMAGE 
imgCountours = img.copy()
imgBigCountours = img.copy()
countours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #finding all countours
cv2.drawContours(imgCountours, countours, -1, (0,255,0), 3)


# FINDING THE BIGGEST OF ALL CONTOURS FOUND
biggest, maxArea = biggestCountour(countours)
if biggest.size != 0:
    biggest = reorder(biggest)
    cv2.drawContours(imgBigCountours, biggest, -1, (0,0,255), 25) # DRAWING THE BIGGEST CONTOUR
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0], [widthImage, 0], [0,heightImage], [widthImage, heightImage]]) # PREPARING POINTS FOR WARPING THE IMAGE 
    
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImage, heightImage)) # WARPING IMAGE IN THIS STEP
    imgDetectedDigits = imgBlank.copy() # THIS WILL BE USED LATER TO PAST DIGITS FOUND FROM SUDOKU ON
    imgWarpColored = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)

    imgSolvedDigits = imgBlank.copy() 
    ret, imgWarpColored = cv2.threshold(imgWarpColored, 127,255,cv2.THRESH_BINARY_INV) # VERY IMPORTANT STEP 
    boxes = splitBoxes(imgWarpColored) # SPLITING THE SUDOKU IMAGE INTO 81 BOXES
    boxes = np.asarray(boxes)
    
    result, result2D = predictSudokuArray(boxes) # FUNCTION FROM pro1 WHICH RESULTS 2D ARRAY AS RESULT
    if result == False:
        print('Some error occurred...\n\n')
        print('Rescan the sudoku image to detect the digits correctly')
    else:
        imgSolvedDigits = imgBlank.copy()
        imgFoundDigits = imgBlank.copy()
        imgFoundDigits = displayNumbers1(imgFoundDigits, result, (0,215,255))
        cv2.imshow('imgFoundDigits', imgFoundDigits)
        cv2.imwrite('imgFoundDigits.png', imgFoundDigits)
        imgSolvedDigits = displayNumbers2(imgSolvedDigits, result, result2D, (0,250,0)) 
        cv2.imshow('imgSolvedDigits', imgSolvedDigits)
        cv2.imwrite('imgSolvedDigits.png', imgSolvedDigits)

        # OVERLAYING SOLUTION 
        pts2 = np.float32(biggest)
        pts1 = np.float32([[0,0],[widthImage,0],[0,heightImage],[widthImage,heightImage]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgInWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImage, heightImage))
        inv_perspective = cv2.addWeighted(imgInWarpColored, 1, img, 0.5, 1)

        cv2.imshow('finalSolutionOverlayed', inv_perspective)
        cv2.imwrite('finalSolutionOverlayed.png', inv_perspective)
        print('\n\nFinal solution obtained:')
        result2D = np.asarray(result2D)
        print(result2D)

else:
    print('sudoku not found!')

cv2.imshow('binaryImgOfUnsolvedSudoku',imgWarpColored)
cv2.imwrite('binaryImgOfUnsolvedSudoku.png',imgWarpColored)
sys.stdout = orig_stdout
f.close()
cv2.waitKey(0)