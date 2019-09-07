## -*- coding: utf-8 -*-
"""
Last amended: 22nd January, 2019
Myfolder: Myfolder: C:\Users\ashok\OneDrive\Documents\python
					/home/ashok/Documents/manifold_learning

Objectives:
    Observe Structure in data using:
            i)   Andrews plots
            ii)  Parrallel axis plots
            iii) Radviz plots
            iv)  t-sne


Data file: titanic (folder datasets)

"""
#%%                                 Call libraries

%reset -f
# 1. Call libraries
## 1. Data manipulation
# 1.1
import pandas as pd
# 1.2
import numpy as np

## 1.3. Handling categorical data
#  1.4 Label Encoder transforms categories ['a','b','c'] to [0,1,2]
from sklearn.preprocessing import  LabelEncoder
#  1.5 Convert to dummy variables
from sklearn.preprocessing import OneHotEncoder


#  1.6 Plotting
import matplotlib.pyplot as plt

## 1.7 Structure in data
# 1.8 Andrews curves plots:
#     https://en.wikipedia.org/wiki/Andrews_plot
from pandas.tools.plotting import andrews_curves

# 1.9 Parallel coordinates plots:
#  https://docs.tibco.com/pub/spotfire/6.5.2/doc/html/para/para_what_is_a_parallel_coordinate_plot.htm
from pandas.tools.plotting import parallel_coordinates

# 1.10 radviz plots:
#   https://cran.r-project.org/web/packages/Radviz/vignettes/single_cell_projections.html
from pandas.tools.plotting import radviz

# 1.11 t-sne
#  https://medium.com/@luckylwk/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
from sklearn.manifold import TSNE

# 1.12
import os


#%%                               Read Data and explore
# 2. Read data and have a look
#os.chdir("C:\\Users\\ashokharnal\\OneDrive\\Documents\\titanic")
os.chdir("E:/lalit/Teaching/ML_DL_March_2019/course/classes/Extra_Classes/manifold_learning/titanic")

os.listdir()
df = pd.read_csv("train.csv.zip")
# 2.1 Observe data
df.shape          # (891,12)
df.columns
df.head(2)

# 3. Drop four columns not needed
df.drop(labels =['PassengerId',
                 'Name',
				 'Ticket',
				 'Cabin'],
				  inplace = True,
				  axis = 1        # Note the
				  )
# 3.1
df.columns

#%%                               Handling categorical variables

df.head(2)

# 4.1 First, encode column 'Embarked' to int [0,1,2]
#  Unfortunately LabelEncoder does not work when the data is
#    null at some places.
sum(df['Embarked'].isnull())     # Two NULLs

# 4.2 We use pandas .map function. It maps
#     values in a Series to values of corresponding
#     keys in a given dictionary
howToMapDict = { 'S' : 0, 'Q' : 1, 'C' : 2}
df['Embarked'] = df['Embarked'].map(howToMapDict)

# 4.3 We will encode 'Sex' column using LabelEncoder.
#     This column has no NULLS
#     Create first a LabelEncoder object
ll = LabelEncoder()
# 4.4 And, then encode
df.iloc[:,2] = ll.fit_transform(df.iloc[:, 2])
df.head()

#%%                               Handling missing data

## 5. Impute missing data now

# 5.1 We will impute missing variables
#      First check which columns have missing values
df.isnull().sum()    # Age and Embarked
df.dtypes            #   Both are float types


# 5.2 Use pandas mean() method to fill in age
df['Age'].fillna(df['Age'].mean(), inplace = True)

# 5.3 We will fill in NULLS in Embarked column with mode
#     Which value is the modal value?
df['Embarked'].value_counts().index[0]      # 0 is the mode

# 5.4 Fill in missing values now
df['Embarked'].fillna(df['Embarked'].value_counts().index[0] , inplace = True)

# 5.5 Check
df['Embarked'].isnull().sum()
df.columns

#%%                              Structure in high dimensional data
#                          Project N-dimensional data set into a simple 2D

#               i)   Andrews curves
#               ii)  Parallel Coordinates
#               iii) Radviz
#               iv) t-sne

## 6. Structure in high dimensional data

# 6.1 Andrews curve
# Let a value x in data X consist of {x1,x2,x3,x4....}. Then a Fourier Series is
#  created as: x1/sqrt(2) + x2sin(t) + x3 (sin2t) + x4 sin(4t) + ..t is varied from
#   -pi to +pi. Thus each data point is plotted as a curve from -pi to +pi
#    These curves have been utilized in fields as different as biology,
#     neurology, sociology and semiconductor manufacturing. Some of their
#      uses include the quality control of products, the detection of outliers
# 6.2 First create a random data.
#     Does random data has any structure?
df1 = pd.DataFrame({ 'a' : [0,1] * 250,
                    'b' : np.random.randn(500),
                    'c' : np.random.randn(500),
                    'd' : np.random.randn(500),
                    'e' : np.random.randn(500),
                    'f' : np.random.randn(500),
                    'g' : np.random.randn(500),
                    'h' : np.random.randn(500),
                    'i' : np.random.randn(500),
                    'j' : np.random.randn(500),
                    'k' : np.random.randn(500),
                    'l' : np.random.randn(500),
                    'm' : np.random.randn(500) })

# 6.3 Plot Andrews curve. Merely a jumble of lines
andrews_curves(df1, "a")

# 6.4 Plot now for titanic data. There appears to be a structure in that
#     there is clear separation between yellow and green lines
andrews_curves(df, "Survived")


# 7.1 Draw parallel coordinates
# 7.2 First with random data
parallel_coordinates(df1, "a")
# 7.3 Next with titanic data
#     Presence of structure is evident
parallel_coordinates(df, "Survived")


# 8. A final multivariate visualization technique pandas
#     has is radviz.
#     In Radviz, each dimension in the dataset is represented by
#      a dimensional anchor, and each dimensional anchor is distributed
#       evenly on a unit circle. Each line in the data set corresponds
#       to a point in the projection, that is linked to every dimensional
#        anchor by a spring. Each spring’s stiffness corresponds to the
#        value for that particular thing in that particular dimension.
#         The position of the point is defined as the point in the 2D space
#          where the spring’s tension is minimum.

# 8.1 First with random data
radviz(df1, "a")   # There is no pulling anywhere
# 8.2 Then with titanic data
radviz(df, "Survived")


# 9. Next is t-sne
tsne= TSNE()
# 9.1 First t-sne of random data
tsne_results_random = tsne.fit_transform(df1.iloc[: , 1:], df1.iloc[: , 0:])
# 9.2 Next, t-sne of titanic data
tsne_results_titanic = tsne.fit_transform(df.iloc[:, 1:], df.iloc[:, 0])

# 10. Plot the two results
# 10.1 First deep copy of random data
df1_tsne = df1.iloc[:, 1:].copy()
# 10.2 Here is X-axis points
df1_tsne['x-tsne'] = tsne_results_random[:,0]
# 10.3 y-axis points
df1_tsne['y-tsne'] = tsne_results_random[:,1]
# 10.4
df1_tsne.shape    # (500,14)

# 10.5
#  List comprehensions provide a concise way to create lists.
#   Common applications are to make new lists where each element
#    is the result of some operations applied to each member of
#      another sequence
# 10.5.1 Expt in list comprehensions
squares = []
for x in range(10):
    squares.append(x**2)

squares

# 10.5.2 Above is same as:
squares = [x**2 for x in range(10)]

# 10.6  Could have also used dictioray mapping and
#       then convert to array
color_random = ['red' if x == 0 else 'green' for x in df1.iloc[:,0 ] ]
color_random[:5]
# 10.6.1
len(color_random)     # 500 colors


"""
List comprehension:
# This syntax works
color1= ['red'  for x in df1.iloc[:,0 ] if x == 0  ]
# But this does not:
color1= ['red'  for x in df1.iloc[:,0 ] if x == 0 else 'green' ]
# It should be:
color1= ['red'  if x == 0 else 'green' for x in df1.iloc[:,0 ]  ]

"""

# 11 t-sne scatter plot for random data
plt.figure()
# 11.1 t-sne plots of random data
plt.scatter(df1_tsne['x-tsne'],df1_tsne['y-tsne'] , color = color_random)
plt.show()

# 12. Next plot the results for titanic dataset
df_tsne = df.iloc[:, 1:].copy()
df_tsne['x-tsne'] = tsne_results_titanic[:,0]
df_tsne['y-tsne'] = tsne_results_titanic[:,1]

# 12.1
color_titanic= ['red' if x == 0 else 'green' for x in df.iloc[:,0 ] ]
len(color_titanic)           # 891

# 12.2  Plot of tsne results with titanic data
plt.figure()
plt.scatter(df_tsne['x-tsne'],df_tsne['y-tsne'] , color = color_titanic)
plt.show()

