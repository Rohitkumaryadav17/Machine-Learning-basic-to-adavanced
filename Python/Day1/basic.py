print("hello ,I am learning machine learning")
# variable and datatype
age=21
height=5.9
print(height,age)
#list
number=[1,2,3,4]
print(number[2])
number.append(5)
print(number)
#loop
for i in range(5):
    print(i)
    #loop through list
    for num in number:
        print(num)
#function
def add(a,b):
    return a+b
print(add(2,4))
#library
import numpy as np
import pandas as pd
#numpy
arr = np.array([1, 2, 3, 4, 5])
print(arr)
#pandas
df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
print(df)

