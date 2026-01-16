# Distance Metrics

Distance metrics play an essential role in machine learning because they can be used to compare the similarity or dissimilarity of two data points. Commonly used for clustering or classification tasks,  distance-based algorithms assume that points in close proximity are more similar to each other. As the proximity of two points is defined by the distance between them, the choice of metric is of essential importance. Depending on the specific needs of your task, you can choose from a variety of different distance metrics. Most frequently used is the Euclidean, which measures the straight-line distance between two points in Euclidean space. Despite its popularity the Euclidean metric should only be used in lower dimensional space (see curse of dimensionality). For higher dimensional space other approaches like Manhattan distance or cosine similarity are more applicable. In this notebook you find an overview of common distance metrics and how to calculate them in python.


```python
# Setup
import numpy as np
```

# Popular distance metrics 


## Manhattan Distance


The Manhattan distance is the sum of absolute differences between the points' coordinates. It is measured along axes at right angles. How the path is taken does not matter, all routes lead to the same distance value. 

<p align="center">
  <img src="images/manhattan.jpg" alt="drawing" width="200"/>
</p>

Image description:
There is a two dimensional square grid with many horizontal and vertical grid lines displayed, with a point marked A in the lower left corner, and a point marked B in the upper right corner. Various paths are traced out between A and B, that take different left/right turns along the grid lines. One turns right only once, another takes a turn at every intersection to reach B from A. The total distance travelled in any case remains the same. All paths only follow the vertical and horizontal directions of the grid lines and stay on them.
End of image description.

### Formula:

$\ Manhattan\: Distance = \sum_{i=1}^k |x_i - y_i|$

$\ Manhattan\: Distance = \sum_{i=1}^k side\: length$

### In python code:


```python
# Example of Manhattan distance 
point_1 = (2, 3, 5)
point_2 = (1, -1, 3)

manhattan_distance = 0
for i in range(3):
    manhattan_distance += np.abs(point_1[i] - point_2[i])

manhattan_distance
```




    7



## Euclidean Distance

The Euclidean distance is defined as the beeline distance between two points. It is calculated from the cartesian coordinates of the points using the pythagorean theorem.

<p align="center">
  <img src="images/euclidean.jpg" alt="drawing" width="200"/>
</p>

Image description:
The same grid as before with many horizontal and vertical grid lines and two points marked A and B in diagonally opposite corners. This time the path shown between A and B is the straight line connecting them, which goes at an angle to the grid lines. 
End of image description.

### Formula:

$\ Euclidean\: Distance = \sqrt{\sum_{i=1}^k |x_i - y_i|^2}$ <br>
$\ Euclidean\: Distance = \sqrt{\sum_{i=1}^k side\:length^2}$

### In python code:


```python
# Example of Euclidean distance
point_1 = (2, 3, 5)
point_2 = (1, -1, 3)

euclidean_distance = 0
for i in range(3):
    euclidean_distance += (point_1[i] - point_2[i])**2
    print(euclidean_distance)

euclidean = np.sqrt(euclidean_distance)

euclidean
```

    1
    17
    21





    4.58257569495584



## Minkowski distance 

The Minkowski distance is a generalization of the Euclidean distance and can be used to calculate the distance between two points in an n-dimensional space.

### Formula:

$\ Minkowski\: Distance = \sqrt[n]{\sum_{i=1}^k |x_i - y_i|^n}$ <br>
$\ Minkowski\: Distance = \sqrt[n]{\sum_{i=1}^k side\:length^n}$

### In python code:


```python
# Example of Minkowski distance with 3 dimensions
point_1 = (2, 3, 5)
point_2 = (1, -1, 3)
p = 3

minkowski_distance = 0
for i in range(3):
    minkowski_distance += np.abs(point_1[i] - point_2[i])**p


minkowski = minkowski_distance**(1/p)

minkowski
```




    4.179339196381232



# Overview 

## Distance Metrics Formulas


**Manhattan Distance is the sum of all side lengths to the first power:**

$\ Manhattan\: Distance = \sum_{i=1}^k |x_i - y_i|$

$\ Manhattan\: Distance = \sum_{i=1}^k side\: length$

**Euclidean Distance is the square root of the sum of all side lengths to the second power:**

$\ Euclidean\: Distance = \sqrt{\sum_{i=1}^k |x_i - y_i|^2}$ <br>
$\ Euclidean\: Distance = \sqrt{\sum_{i=1}^k side\:length^2}$

**Minkowski Distance with a value of 3 would be the cube root of the sum of all side lengths to the third power:**

$\ Minkowski\: Distance = \sqrt[3]{\sum_{i=1}^k |x_i - y_i|^3}$ <br>
$\ Minkowski\: Distance = \sqrt[3]{\sum_{i=1}^k side\:length^3}$

**Minkowski Distance with a value of 5:**


$\ Minkowski\: Distance = \sqrt[5]{\sum_{i=1}^k |x_i - y_i|^5}$ <br>
$\ Minkowski\: Distance = \sqrt[5]{\sum_{i=1}^k side\:length^5}$
