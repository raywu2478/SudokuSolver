# SudokuSolver
Locates, recognizes, and solves a sudoku puzzle from an image

Requires openCV and Numpy

## Process
### Finding the puzzle
First convert to B&W image for easier contour recognition. 
Then find the largest contour in the image. Here I am assuming the main focus of the image should is the puzzle.
By drawing a bounding box around the largest contour, the puzzle is now located.
Knowing the standard sudoku puzzle is square, I warp the bounding box to a square. This should be a head on view of the puzzle.
### Recognize the numbers and training a classifier
Next the puzzle is evenly divided into 9x9 squares since the numbers should be spaced equally.
The extracted numbers can be used to train a classifier of choice. Here I used random forest calssifier from scikit-learn.
### Solving the puzzle
The sudoku puzzle is then solved recursively.
