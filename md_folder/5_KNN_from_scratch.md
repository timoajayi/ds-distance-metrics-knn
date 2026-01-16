# KNN from Scratch

Using the KNN algorithm from scikit-learn is quite simple. When you use it, you don't even need to know how KNN works and classifies the data points. The goal should be to really understand what you are doing. This is the only way to make sure you understand when something doesn't go the way you expect or want it to. Many people remember things best when they implement them themselves. So, let's build our own K-Nearest-Neighbors classifier!

**The purpose of this notebook is to help you remember the steps necessary to classify samples with KNN.**

To test if your code works, you can use the Iris dataset as a data example.
Let's make a plan and break this big task into smaller steps!


1. What information and data does the algorithm need to train and predict the classes of new instances?
This will be the input for our function! 

2. calculate the distance between the test point and each existing data point in the training data.
3. determine the nearest k neighbors.
4. make predictions based on these neighbors.

You have already implemented a function to calculate the distance between points, which will now come in handy.

A good way to get started, is to ignore the syntax and just write in simple text what you want your program to do aka **write pseudo-code**. You can then start to build out some of the structure. What variables are you going to need? What kinds of logic? 
Knowing where you’re going can help you make fewer mistakes as you’re trying to get there.

Note that for large data sets, the algorithm can take very long to classify because it has to calculate the distance between the test point and every other point in the data!

You can check if your pseudo-code contains all necessary steps afterwards, when scrolling down to "KNN algorithm from scratch" where you find an example of a knn pseudo-code.

## Import and Setup


```python
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
```


```python
# Load data
df = pd.read_csv('data/iris.csv')
```


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    Cell In[2], line 2
          1 # Load data
    ----> 2 df = pd.read_csv('data/iris.csv')


    File ~/Documents/Encounters/Student-repos/ds-distance-metrics-knn/.venv/lib/python3.11/site-packages/pandas/io/parsers/readers.py:912, in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
        899 kwds_defaults = _refine_defaults_read(
        900     dialect,
        901     delimiter,
       (...)
        908     dtype_backend=dtype_backend,
        909 )
        910 kwds.update(kwds_defaults)
    --> 912 return _read(filepath_or_buffer, kwds)


    File ~/Documents/Encounters/Student-repos/ds-distance-metrics-knn/.venv/lib/python3.11/site-packages/pandas/io/parsers/readers.py:577, in _read(filepath_or_buffer, kwds)
        574 _validate_names(kwds.get("names", None))
        576 # Create the parser.
    --> 577 parser = TextFileReader(filepath_or_buffer, **kwds)
        579 if chunksize or iterator:
        580     return parser


    File ~/Documents/Encounters/Student-repos/ds-distance-metrics-knn/.venv/lib/python3.11/site-packages/pandas/io/parsers/readers.py:1407, in TextFileReader.__init__(self, f, engine, **kwds)
       1404     self.options["has_index_names"] = kwds["has_index_names"]
       1406 self.handles: IOHandles | None = None
    -> 1407 self._engine = self._make_engine(f, self.engine)


    File ~/Documents/Encounters/Student-repos/ds-distance-metrics-knn/.venv/lib/python3.11/site-packages/pandas/io/parsers/readers.py:1661, in TextFileReader._make_engine(self, f, engine)
       1659     if "b" not in mode:
       1660         mode += "b"
    -> 1661 self.handles = get_handle(
       1662     f,
       1663     mode,
       1664     encoding=self.options.get("encoding", None),
       1665     compression=self.options.get("compression", None),
       1666     memory_map=self.options.get("memory_map", False),
       1667     is_text=is_text,
       1668     errors=self.options.get("encoding_errors", "strict"),
       1669     storage_options=self.options.get("storage_options", None),
       1670 )
       1671 assert self.handles is not None
       1672 f = self.handles.handle


    File ~/Documents/Encounters/Student-repos/ds-distance-metrics-knn/.venv/lib/python3.11/site-packages/pandas/io/common.py:859, in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
        854 elif isinstance(handle, str):
        855     # Check whether the filename is to be opened in binary mode.
        856     # Binary mode does not support 'encoding' and 'newline'.
        857     if ioargs.encoding and "b" not in ioargs.mode:
        858         # Encoding
    --> 859         handle = open(
        860             handle,
        861             ioargs.mode,
        862             encoding=ioargs.encoding,
        863             errors=errors,
        864             newline="",
        865         )
        866     else:
        867         # Binary mode
        868         handle = open(handle, ioargs.mode)


    FileNotFoundError: [Errno 2] No such file or directory: 'data/iris.csv'



```python
# Defining X and y 
```


```python
# Train test split

```

## Distance Metrics

As already explained, KNN assigns a class to the test point based on the majority class of  K  nearest neighbors. In general, euclidean distance is used to find nearest neighbors, but other distance metrics can also be used.

As the dimensionality of the feature space increases, the euclidean distance often becomes problematic due to the curse of dimensionality (discussed later).

In such cases, alternative vector-based similarity measures (dot product, cosine similarity, etc) are used to find the nearest neighbors. This transforms the original metric space into one more amicable to point-to-point measurements.

Another distance measure that you might consider is [Mahalanobis distance](https://en.wikipedia.org/wiki/Mahalanobis_distance). Mahalanobis distance attempts to weight features according to their probabilities. On some data sets that may be important.

In general, it's probably a good idea to normalize the data at a minimum. Here's a link to the scikit-learn scaling package: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html . You have to be a little circumspect about employing any technique where the answers change with scaling.


```python
# Implemented own distance function 

```

## KNN Algorithm from scratch


Remember the steps:

1. What information and data does the algorithm need to train and predict the classes of new instances?
This will be the input for our function! 

2. calculate the distance between the test point and each existing data point in the training data.
3. determine the nearest k neighbors.
4. make predictions based on these neighbors.

Hopefully you have already thought of your gameplan, also called pseudo-code. If so, you can compare it to this one:
```
INPUT: X_train, y_train, X_test, k
FOR each object_to_predict in X_test:
    FOR each training_point, index in X_train:
        calculate distance d between object_to_predict and training_point
        store d and index
    SORT distances d in increasing order
    take first k items, get indices of those
    calculate most common class of points at indices in y_train (prediction)
    store prediction
RETURN list of predictions
````

Time to code!
Don't forget that it's good practice to document your own code! This way you can later understand what the purpose of each step was.
Maybe you can even use your pseudo code as documentation :)


```python
# Your code
```

## Comparison with sklearn knn implementation

That will be interesting! Check out how your implementation performs in comparison to the one of sklearn!
You can check the confusion matrix and the accuracy score of both algorithms.
If you want, you can check which algorithm is faster!


```python
# Your code
```
