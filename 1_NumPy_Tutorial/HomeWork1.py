import numpy as np
import time
import random

# name : Bhushan Jagtap

# - Fill in the code below the comment Python and NumPy same as in example.
# - Follow instructions in document.
###################################################################
# Example: Create a zeros vector of size 10 and store variable tmp.
# Python
pythonStartTime = time.time()
tmp_1 = [0 for i in range(10)]
pythonEndTime = time.time()

# NumPy
numPyStartTime = time.time()
tmp_2 = np.zeros(10)
numPyEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))


z_1 = None
z_2 = None
################################################################
# 1. Create a zeros array of size (3,5) and store in variable z.
# Python

print("\n1.Create a zeros array of size (3,5) and store in variable z.")
pythonStartTime = time.time()
z_1 = [[0 for x in range(5)] for y in range(3)]
pythonEndTime = time.time()

# NumPy
numPyStartTime = time.time()
z_2 = np.zeros((3,5), dtype='int64')
numPyEndTime = time.time()

# print
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))
print("Result :", np.sum(z_1 == z_2))

#################################################
# 2. Set all the elements in first row of z to 7.
# Python
print("\n2. Set all the elements in first row of z to 7.")

pythonStartTime = time.time()

len1 = len(z_1[0])
for i in range(len1):
    z_1[0][i] = 7

pythonEndTime = time.time()
# NumPy

numPyStartTime = time.time()
z_2[0,:] = 7
numPyEndTime = time.time()

print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))
print("Result :", np.sum(z_1 == z_2))

#####################################################
# 3. Set all the elements in second column of z to 9.
print("\n3. Set all the elements in second column of z to 9.")

# Python

pythonStartTime = time.time()
len1 = len(z_1)
for i in range(len1):
    z_1[i][1] = 9
pythonEndTime = time.time()

# NumPy

numPyStartTime = time.time()
z_2[0:3, 1] = 9
numPyEndTime = time.time()

print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))
print("Result :", np.sum(z_1 == z_2))

#############################################################
# 4. Set the element at (second row, third column) of z to 5.
print("\n4. Set the element at (second row, third column) of z to 5.")
# Python

pythonStartTime = time.time()
z_1[1][2] = 5
pythonEndTime = time.time()

# NumPy
numPyStartTime = time.time()
z_2[1,2] = 5
numPyEndTime = time.time()


print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))
print("Result :", np.sum(z_1 == z_2))

##############
print(z_1)
print(z_2)
##############


x_1 = None
x_2 = None
##########################################################################################
# 5. Create a vector of size 50 with values ranging from 50 to 99 and store in variable x.
print("\n5. Create a vector of size 50 with values ranging from 50 to 99 and store in variable x.")
# Python

pythonStartTime = time.time()
x_1 = [i for i in range(50, 100)]
pythonEndTime = time.time()

# NumPy
numPyStartTime = time.time()
x_2 = np.array(range(50, 100))
numPyEndTime = time.time()

print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))
print("Result :", np.sum(x_1 == x_2))

##############
print(x_1)
print(x_2)
##############


y_1 = None
y_2 = None
##################################################################################
# 6. Create a 4x4 matrix with values ranging from 0 to 15 and store in variable y.
# Python
print("\n6. Create a 4x4 matrix with values ranging from 0 to 15 and store in variable y.")

pythonStartTime = time.time()
y_1 =[]
for i in range(4):
    y_1.append([])
    for j in range(4):
        y_1[i].append((i * 4)+j)
pythonEndTime = time.time()

# NumPy

numPyStartTime = time.time()
y_2 = np.array(range(0,16)).reshape(4,4)
numPyEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))
print("Result :", np.sum(y_1 == y_2))

##############
print(y_1)
print(y_2)
##############


tmp_1 = None
tmp_2 = None
####################################################################################
# 7. Create a 5x5 array with 1 on the border and 0 inside and store in variable tmp.

print("\n7. Create a 5x5 array with 1 on the border and 0 inside and store in variable tmp.")

# Python
pythonStartTime = time.time()
tmp_1 = [[0 for x in range(5)] for y in range(5)]
for i in range(5):
    for j in range(5):
        tmp_1[0][j] = 1
        tmp_1[4][j] = 1
        tmp_1[j][0] = 1
        tmp_1[j][4] = 1
pythonEndTime = time.time()

# NumPy
numPyStartTime = time.time()
tmp_2 = np.ones((5,5))
tmp_2[1:-1,1:-1] = 0
numPyEndTime = time.time()

print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))
print("Result :", np.sum(tmp_1 == tmp_2))

##############
print(tmp_1)
print(tmp_2)
##############


a_1 = None; a_2 = None
b_1 = None; b_2 = None
c_1 = None; c_2 = None

# 8. Generate a 50x100 array of integer between 0 and 5,000 and store in variable a.
# Python
print("\n8.PYTHON")
a_1 = [[0 for i in range(100)] for j in range(50)]
k=0
for i in range(50):
    for j in range(100):
        a_1[i][j]=k
        k+=1
print(a_1)
# NumPy
print("\n8.NUMPY")
a_2 = np.array(np.arange(0,5000).reshape(50,100))
print(a_2)

###############################################################################################
# 9. Generate a 100x200 array of integer between 0 and 20,000 and store in variable b.
# Python
print("\n9.PYTHON")
b_1 = [[0 for i in range(200)] for j in range(100)]
k=0
for i in range(100):
    for j in range(200):
        b_1[i][j]=k
        k+=1
print(b_1)
# NumPy
print("\n9.NUMPY")
b_2 = np.array(np.arange(0,20000).reshape(100,200))
print(b_2)


#####################################################################################
# 10. Multiply matrix a and b together (real matrix product) and store to variable c.
# Python
print("\n10.PYTHON")
c_1 = [[0 for i in range(200)] for y in range(50)]
for i in range(len(a_1)):
    for j in range(len(b_1[0])):
        for k in range(len(b_1)):
            c_1[i][j] += a_1[i][k] * b_1[k][j]

for r in c_1:
 print(r)

# NumPy
print("\n10.NUMPY")
c_2 = np.dot(a_2,b_2)
print(c_2)

print("BHUSHAN :", np.sum(c_1 == c_2))


d_1 = None; d_2 = None
################################################################################

# 11. Normalize a 3x3 random matrix ((x-min)/(max-min)) and store to variable d.
# Python

print ("\n11. Normalize a 3x3 random matrix ((x-min)/(max-min)) and store to variable d.")

pythonStartTime = time.time()
d_1 =[]
for i in range(3):
    d_1.append([])
    for j in range(3):
        d_1[i].append(random.randrange(0, 100))


min1 = max1 = d_1[0][0]

for i in range(3):
    for j in range(3):
        min1 = min(min1,d_1[i][j])
        max1 = max(max1,d_1[i][j])


for i in range(3):
    for j in range(3):
        d_1[i][j] = (d_1[i][j] - min1)/(max1 - min1)

pythonEndTime = time.time()

# NumPy

numPyStartTime = time.time()

d_2 = np.random.random((3,3))
min1 = d_2.min()
max1 = d_2.max()
d_2 = (d_2 - min1)
d_2 = d_2 / (max1 - min1)

numPyEndTime = time.time()

print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))


##########
print(d_1)
print(d_2)
#########


################################################
# 12. Subtract the mean of each row of matrix a.
print ("\n12. Subtract the mean of each row of matrix a.")

# Python
pythonStartTime = time.time()
row_mean_of_a = [0 for i in range(50)]

for i in range(50):
    for j in range(100):
        row_mean_of_a[i] += a_1[i][j]
    row_mean_of_a[i] /= 100.0

for i in range(50):
    for j in range(100):
        a_1[i][j] -= row_mean_of_a[i]

pythonEndTime = time.time()
# NumPy

numPyStartTime = time.time()
a_2 = a_2 - a_2.mean(axis=1, keepdims=True)
numPyEndTime = time.time()

print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))


###################################################
# 13. Subtract the mean of each column of matrix b.
# Python
print(np.sum(b_1 == b_2))
print ("\n13. Subtract the mean of each column of matrix b")

pythonStartTime = time.time()
col_mean_of_b = [0 for i in range(200)]
for i in range(100):
    for j in range(200):
        col_mean_of_b[j] += b_1[i][j]

for j in range(200):
    col_mean_of_b[j] /= 100.0

for i in range(100):
    for j in range(200):
        b_1[i][j] -= col_mean_of_b[j]

pythonEndTime = time.time()

# NumPy

numPyStartTime = time.time()
b_2 = b_2 - b_2.mean(axis=0, keepdims=True)
numPyEndTime = time.time()

print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))

################
print(np.sum(a_1 == a_2))
print(np.sum(b_1 == b_2))
################

e_1 = None; e_2 = None
###################################################################################
# 14. Transpose matrix c, add 5 to all elements in matrix, and store to variable e.
# Python

print("\n14. Transpose matrix c, add 5 to all elements in matrix, and store to variable e.")
pythonStartTime = time.time()
e_1 = [[0 for i in range(50)] for j in range(200)]
for i in range(50):
    for j in range(200):
        e_1[j][i] = c_1[i][j] + 5
pythonEndTime = time.time()

# NumPy
numPyStartTime = time.time()
e_2 = np.transpose(c_2) + 5
numPyEndTime = time.time()

print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))
##################
print ("result :", np.sum(e_1 == e_2))
##################


#####################################################################################
# 15. Reshape matrix e to 1d array, store to variable f, and print shape of f matrix.
# Python

print("\n15. Reshape matrix e to 1d array, store to variable f, and print shape of f matrix.")
pythonStartTime = time.time()
f_1 = [0 for i in range(10000)]
for i in range(200):
    for j in range(50):
        f_1[i*10+j] = e_1[i][j]
pythonEndTime = time.time()

# NumPy
numPyStartTime = time.time()
f_2 = np.reshape(e_2, 10000)
numPyEndTime = time.time()

print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))
print ("result :", np.sum(e_1 == e_2))
print("size of f_1 : python", len(f_1))
print("size of f_2 : numpy", f_2.shape)
