# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn import datasets

#loading the datasets
iris = datasets.load_iris()

# Plot the first three PCA dimensions
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
y=iris.target
X_reduced = decomposition.PCA(n_components=3).fit_transform(iris.data)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
#Add Title to plot
ax.set_title("First three PCA directions")

#Set x-axis label
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])

#set y-axis label
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])

#set z axis label
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

#Show the plot
plt.show()