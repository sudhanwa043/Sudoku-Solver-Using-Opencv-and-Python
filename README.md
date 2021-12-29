# Sudoku-Solver-Using-Opencv-and-Python
It is a project in which a sudoku image is taken as input as a problem statement and same image is provided along with solution embedded on it as solution.

**Version of python used:**
python 3.9.5

**Required Libraries:**
* OpenCv 
* numpy
* tensorflow
* sys
* matplotlib

## Overview of the inside mechanism
The entire working of the algorithm can be split into 4 major parts
1. **Preparing image easier to understand by the computer:** Scanned image containing sudoku grid is taken as input. The image is scanned for finding biggest contour having 4 edges. Co-ordinates of the biggest contour are noted. The image is then cropped to fit the sudoku grid. Color of the image is changed to gray(1 channel) from BGR image type(3-channels).
2. **Predicting digits in Sudoku:** Then the wrapped sudoku image is divided into 81 images by spliting on 9x9 grid lines. That is each digit is converted into image and all such images are stored in an array. The array is passed to digit classification models to predict digits(if any) on it. The information given by the model is then processed afterwhich a final 2-Dimensional grid is returned containing given clues in sudoku.
3. **Solving the sudoku and overlaying the solution on the input image:** The grid is passed to sudoku solving function which works on the famous algorithm of backtracking, which return the same grid filled with the solution. Solved digits are then extracted from the solution and rendered on the original image. The imaged is then unwrapped and is passed as the final solution
4. **Training digit classification model to predict digits:** Digit classification model used here uses the mnist data set containing 70,000 images handwritten digits. The model is trained on 5 layers neural network, runned 10 times on training data set to optimise accuracy of prediction. It is then saved .h5 extension which can be used directly by other programs to predict digits on images. 

### Sample input image of the sudoku:
![sudokuResource](https://user-images.githubusercontent.com/83408653/147630795-3b58c37b-fff2-49e8-9545-f1dcd90622b6.jpeg)

### Binary version of image easily understandable by computer:
![binaryImgOfUnsolvedSudoku](https://user-images.githubusercontent.com/83408653/147630841-725348d2-a84c-4d35-9ba3-6668838d6402.png)

### Predicted digits of sudoku:
![imgFoundDigits](https://user-images.githubusercontent.com/83408653/147630874-c31762ac-a871-4739-80cd-86f5ae930354.png)

### Solved digits of sudoku:
![imgSolvedDigits](https://user-images.githubusercontent.com/83408653/147630909-b75b7df5-114d-4948-b3b6-885ccbfb16a0.png)

### Solved digits overlayed on the main input image:
![finalSolutionOverlayed](https://user-images.githubusercontent.com/83408653/147630938-b3065114-2e96-45e9-8567-4f62e2872899.png)

