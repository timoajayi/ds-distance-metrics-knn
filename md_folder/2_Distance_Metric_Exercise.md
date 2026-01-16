# Distance metrics exercise

Complete the `distance()` function.


This function should take in 3 arguments:

- a tuple or array that describes a vector in n-dimensional space.

- a tuple or array that describes a vector in n-dimensional space (both tuples should be of same length!)

- an argument which tells us the norm to calculate the vector space (if set to 1, the result will be Manhattan, while 2 will calculate Euclidean distance)

Since Euclidean distance is the most common distance metric used, this function should default to using c=2 if no value is set for c.

Include a parameter called verbose which is set to True by default. If true, the function should print out if the distance metric returned is a measurement of Manhattan, Euclidean, or Minkowski distance.


This function should implement the Minkowski distance equation and return the result.


> NOTE: Remember that for Manhattan Distance, you need to make use of np.abs() to get the absolute value of the distance for each dimension, since we don't have the squaring function to make this positive for us!

> HINT: Use np.power() as an easy way to implement both squares and square roots. np.power(a, 3) will return the cube of a, while np.power(a, 1/3) will return the cube root of 3. For more information on this function, see the [NumPy documentation](https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.power.html)!


```python
import numpy as np


def distance():
    pass

pt1 = (3, 5)
pt2 = (1, 9)
print(distance(pt1, pt2))
print(distance(pt1, pt2, c=1)) 
print(distance(pt1, pt2, c=3)) 
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Cell In[1], line 9
          7 pt1 = (3, 5)
          8 pt2 = (1, 9)
    ----> 9 print(distance(pt1, pt2))
         10 print(distance(pt1, pt2, c=1)) 
         11 print(distance(pt1, pt2, c=3)) 


    TypeError: distance() takes 0 positional arguments but 2 were given


## Testing the function

Calculate the Euclidean distance between the following points in 6-dimensional space:

Point 1: (-2, -5.8, 14, 15, 7, 9)

Point 2: (3, -9.2, -33, -21, 7, 9)

------

Calculate the Manhattan distance between the following points in 9-dimensional space:

Point 1: [0, 0, 0, 7, 16, 2, 0, 1, 12, ]
Point 2: [21, -1, 35, 17, 14, 73, -22, 33, 3, ]

--------

Calculate the Minkowski distance with a norm of 3.5 between the following points:

Point 1: (-32, 47, 223.4) Point 2: (883, 34, 199.5)


```python

```
