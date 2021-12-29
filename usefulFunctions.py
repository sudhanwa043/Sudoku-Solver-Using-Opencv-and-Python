import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sudokuSolvingAlgo import *

def preProcess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converting image to gray scale
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1) #adding gaussian blur
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2) #applying adaptive threshold
    ret, thresh1 = cv2.threshold(imgThreshold, 127,255,cv2.THRESH_BINARY) #converting image to binary
    return thresh1


def biggestCountour(contours):
    biggest = np.array([])
    max_area=0
    for i in contours:
        area = cv2.contourArea(i)
        if area>50:
            peri = cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i,0.02*peri,True)
            if area>max_area and len(approx)==4:
                biggest = approx
                max_area = area
    return biggest, max_area


def reorder(myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2), dtype = np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew


def splitBoxes(img):
    rows = np.vsplit(img,9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r,9)
        for box in cols:
            boxes.append(box)
    return boxes

def initializePredictionModel():
    model=load_model('model.h5')
    return model

def prepareImage(image):
    img = np.asarray(image)
    img = img[4:img.shape[0]-4, 4:img.shape[1]-4]
    img = cv2.resize(img, (28,28))
    img = img/255.0
    img = img.reshape(28,28)
    return img

def listTo2D(result):
    finalResult = []
    for t in range(0,9):
        row = []
        for tt in range(0,9):
            row.append(result[9*t+tt])
        finalResult.append(row)
    return finalResult

def displayNumbers1(img, result, color = (255,0,0)):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for x in range(0,9):
        for y in range(0,9):
            if result[y][x] !=0 :
                cv2.putText(img, str(result[y][x]), (x*secW + int(secW/2)-10, int((y+0.8)*secH)), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                2, color, 2, cv2.LINE_AA)
    return img

def displayNumbers2(img, result, result2D, color = (0,255,0)):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for x in range(0,9):
        for y in range(0,9):
            if result[y][x] ==0 :
                cv2.putText(img, str(result2D[y][x]), (x*secW + int(secW/2)-10, int((y+0.8)*secH)), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                2, color, 2, cv2.LINE_AA)
    return img

def predictSudokuArray(boxes):
    model = load_model('model.h5')
    result = []
    probability = []
    intensity = []
    value = []
    for box in boxes:
        box = prepareImage(box)
        img = np.asarray([box])
        prediction = model.predict(img)
        predictedValue = np.argmax(prediction)
        probabilityOfValue = np.amax(prediction)

        if cv2.countNonZero(box)<70 and predictedValue>1:
            result.append(0)
        elif probabilityOfValue<0.97:
            result.append(0)
        else:
            result.append(predictedValue)
        value.append(predictedValue)
        probability.append(round(probabilityOfValue,2))
        intensity.append(cv2.countNonZero(box))
    
    value = np.asarray(value)
    value = value.reshape(9,9)
    print('\n\nRaw Predicted Values:')
    print(value)

    probability = np.asarray(probability)
    probability = probability.reshape(9,9)
    print('\n\nProbability of predicted values:')
    print(probability)

    intensity = np.asarray(intensity)
    intensity = intensity.reshape(9,9)
    print('\n\nNumber of enlightned pixels in each box of image:')
    print(intensity)

    #result = sudoku(result)
    result2D = listTo2D(result)
    sudoku(result2D) # result2D FILLED WITH SOLUTION
    result = listTo2D(result)
    # print('\n\nTotal solutions:')
    # print(result2D)
    return result, result2D

