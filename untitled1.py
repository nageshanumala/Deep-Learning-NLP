"""
Last amended: 23rd January, 2019


References:
a. In-Depth: Manifold Learning
   https://jakevdp.github.io/PythonDataScienceHandbook/05.10-manifold-learning.html
b. sklearn: Manifold Learning
   https://scikit-learn.org/stable/modules/manifold.html


=========================================
Comparison of Manifold Learning methods
=========================================

An illustration of dimensionality reduction on
the S-curve dataset with various manifold learning
methods.

Note that the purpose of the MDS is to find a low-dimensional
representation of the data (here 2D) in which the distances
respect well the distances in the original high-dimensional
space, unlike other manifold-learning algorithms, it does not
seeks an isotropic representation of the data in the low-dimensional
space.
# Author: Jake Vanderplas -- <vanderplas@astro.washington.edu>



Objectives:
	Learn to draw:
			i)   Multidimensional Scaling (MDS) plots
			ii)  Isomap plots
			iii) Local Linear embedding (LLE) plots
			iv)  Spectral clustering plots
			 v)	 t-sne plots 


"""

## 1.0 Call libraries

%reset -f

# 11
from time import time
# 1.2
import matplotlib.pyplot as plt
# 1.3 For 3d plot
from mpl_toolkits.mplot3d import Axes3D
# 1.4 manifold routines and datasets
from sklearn import manifold, datasets


## 2.0 Generate data
n_points = 1000
# 2.1 Generate points and their respective colors
X, color = datasets.make_s_curve(n_points, random_state=0)

# 2.2
X.shape       # (1000,3): Coordinates of each of 1000 points on 3d axis
color.shape   # (1000,): For each of 1000 points, there is a color-label

# 2.3
X[:2]
color[:10]

# 3. We will need following constants
#    in LLE and ISOMAP algorithms
n_neighbors = 10
n_components = 2


## 4.0 Plot initia 3-Data
fig = plt.figure(figsize=(10,10))
# 4.1 It is a 3-d plot
ax1 = fig.add_subplot(231, projection='3d')
# 4.2 Plot all three axis
ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=color)



## 5. MDS
# MDS has no concept of neighbors
# 5.1
start = time()
mds = manifold.MDS(
                   n_components,
                   max_iter=100,
                   n_init=1
                   )
# 5.2 Learn and transform
Y = mds.fit_transform(X)
end = time()
print(f"MDS: {(end - start): .2f} sec")

# 5.3
ax4 = fig.add_subplot(234)
ax4.scatter(Y[:, 0], Y[:, 1], c=color)

# 5.4 For f-string formatting. pl refer:
#     Ref: https://stackoverflow.com/a/50340297
plt.title(f"MDS {(end-start): .2f} sec " )




## 6. Isomap

# 6,1
start = time()
Y = manifold.Isomap(n_neighbors,
                    n_components
                    ).fit_transform(X)
end = time()
print(f"Isomap {(end-start): .2f} sec")

# 6.2 Plot the transformed data
ax3 = fig.add_subplot(233)
ax3.scatter(Y[:, 0], Y[:, 1], c=color)
plt.title(f"Isomap {(end-start): .2f} sec)" )




## 7.0 Local Linear Embedding

# 7.1 
start = time()
Y = manifold.LocallyLinearEmbedding(n_neighbors,
                                    n_components,
                                    eigen_solver='auto',
                                    method='standard'
                                    ).fit_transform(X)
end = time()

# 7.2 
print(f"LLE time {(end-start): .2f} sec")

# 7.3 Plot the transformed 2d data now
ax2 = fig.add_subplot(232)
ax2.scatter(Y[:, 0], Y[:, 1], c=color)
plt.title(f"LLE {end-start} sec " ) 




## 8.0 Spectral Embedding

# 8.1
start = time()
se = manifold.SpectralEmbedding(
                               n_components=n_components,
                               n_neighbors=n_neighbors
                               )
Y = se.fit_transform(X)
end = time()
print(f"SpectralEmbedding: {(end-start): .2f} sec" )

# 8.2
ax5 = fig.add_subplot(235)
ax5.scatter(Y[:, 0], Y[:, 1], c=color)
plt.title(f"SpectralEmbedding {(end - start): .2f}" )

## 9. t-sne

# 9.1
start = time()
tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
Y = tsne.fit_transform(X)
end = time()
print(f"t-SNE: {(end-start): .2f} sec")

# 9.2
ax6 = fig.add_subplot(2, 3, 6)
ax6.scatter(Y[:, 0], Y[:, 1], c=color)
plt.title(f"t-SNE {(end - start): .2f} sec" )
plt.show()
##############################################################################3
