# Homework 1: Python and NumPy exercises

### Description

In this homework you will practice your coding skill in Python with NumPy package. The goal of this assignment is to help you get familiar with NumPy functions and write program in
Python.

### Instruction

In this homework, you need to write a program in Python that performs the 15 steps listed below. For each step, you need to write 2 version of codes. One uses pure Python and the other
uses NumPy package. You will be asked to collect the run time information of the two versions of program, analyze and compare their speed in your report.

Example: Create a zeros vector of size 10 and store variable tmp.

Python answer: tmp_1 = [0 for i in range(10)]

NumPy answer: tmp_2 = np.zeros(10)

### Steps:

1. Create a zeros array of size (3,5) and store in variable z.
2. Set all the elements in first row of z to 7.
3. Set all the elements in second column of z to 9.
4. Set the element at (second row, third column) of z to 5.
5. Create a vector of size 50 with values ranging from 50 to 99 and store in variable x.
6. Create a 4x4 matrix with values ranging from 0 to 15 and store in variable y.
7. Create a 5x5 array with 1 on the border and 0 inside.
8. Generate a 50x100 array of integer between 0 and 5,000 and store in variable a.
9. Generate a 100x200 array of integer between 0 and 20,000 and store in variable b.
10. Multiply matrix a and b together (real matrix product) and store to variable c.
11. Normalize a 3x3 random matrix ((x-min)/(max-min)) and store to variable d.
12. Subtract the mean of each row of matrix a.
13. Subtract the mean of each column of matrix b.
14. Transpose matrix c, add 5 to all elements in matrix, and store to variable e.
15. Reshape matrix e to 1d array, store to variable f, and print shape of f matrix.
