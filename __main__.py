import cv2
import locate
import slice
import read

# for i in range(184, 219):
#     print i
#     puzzle = locate.find_pizzle(cv2.imread('sudoku' + str(i) + '.jpg'))
#     slice.gather(puzzle)

puzzle = locate.find_pizzle(cv2.imread('sudoku2.jpg'))
slice.gather(puzzle)
