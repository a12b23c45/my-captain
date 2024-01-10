 # To check the version of python 

import sys 
print('python:{}'.format(sys.version))
import scipy
print('scipy:{}'.format(scipy._version_))
import numpy
print('numpy:{}'.format(numpy._version_))
import matplotlib
print('matplotlib:{}'.format(matplotlib._version_))
import pandas
print('pandas:{}'.format(pandas._version_))
import sklearn
print('sklearn:{}'.format(sklearn._version_))

#importing various plots and variables

import pandas
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier

# loading the dataset 
url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# dimensions of the dataset
print(dataset.shape)

#taking a peek at the data
print(dataset.head(20))

#statistical summary
print(dataset.describe())
