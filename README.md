[![Shipping files](https://github.com/neuefische/ds-distance-metrics-knn/actions/workflows/workflow-02.yml/badge.svg?branch=main&event=workflow_dispatch)](https://github.com/neuefische/ds-distance-metrics-knn/actions/workflows/workflow-02.yml)

# Distance Metrics & K-Nearest Neighbors 

In this repository, we practice different
distance metrics in Python and have a look at the k-nearest neighbors algorithm.

## The way to success:

Please work together as **Pair-Programmers** through all the notebooks
in this particular order:

1. [Distance Metric in Python](1_Distance_Metric_Python.ipynb)
2. [Distance Metric Exercise](2_Distance_Metric_Exercise.ipynb)
3. [KNN in sklearn](3_KNN_sklearn.ipynb)
4. [KNN Exercise](4_KNN_Exercise.ipynb)
5. [KNN from scratch](5_KNN_from_scratch.ipynb)

The first notebook will teach you how different 
distance metrics are calculated in python.
In the second notebook, you will write your own 
function to use your new knowledge about 
distance metrics practically in python.

The fourth notebook will show you how to implement
KNN with sklearn, while you can try this by yourself 
directly in the fifth notebook.
In the sixth notebook, you have the chance to create 
your own KNN Algorithm!

## Set up your Environment

Please make sure you have forked the repo and set up a new virtual environment. For this purpose you can use the following commands:

The added [requirements file](requirements.txt) contains all libraries and dependencies we need to execute the  Distance Metrics & K-Nearest Neighbors notebooks.

*Note: If there are errors during environment setup, try removing the versions from the failing packages in the requirements file. M1 shizzle.*

### **`macOS`** type the following commands : 


- Install the virtual environment and the required packages by following commands:

    ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
### **`WindowsOS`** type the following commands :

- Install the virtual environment and the required packages by following commands.

   For `PowerShell` CLI :

    ```PowerShell
    pyenv local 3.11.3
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

    For `Git-Bash` CLI :
    ```
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/Scripts/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```
