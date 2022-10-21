# import image module
from IPython.display import Image
  

# ------ Display confusion matrix function ------
def matrix():

# get the image
    return Image(url="confusion_matrix.png", width=920, height=474)



# ------ Print standard import functions for quick use ------
def imports():
     print('''
# standardized modules
import seaborn as sns
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import train_test_split

# Decision Tree and Model Evaluation Imports
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

#my modules
import acquire as ac
import prepare as pp
import helper

#turn off the red warnings
import warnings
warnings.filterwarnings("ignore")
''')

# ------ Useful information for making a decision tree ------
def tree():
    print('''
_______________________________________________________________
|                              DF                             |
|-------------------|-------------------|---------------------|
|       Train       |       Validate    |          Test       |
|-------------------|-------------------|-----------|---------|
| x_train | y_train |   x_val  |  y_val |   x_test  |  y_test |
|-------------------|-------------------|-----------|---------|
```
* 1. tree_1 = DecisionTreeClassifier(max_depth = 5)
* 2. tree_1.fit(x_train, y_train)
* 3. predictions = tree_1.predict(x_train)
* 4. pd.crosstab(y_train, predictions)
* 5. val_predictions = tree_1.predict(x_val)
* 6. pd.crosstab(y_val, val_predictions)
''')