""" This Class is the starting point of program and driver class
@author Sayali Kudale
"""
import pca as pca
import numpy as np
import matplotlib.pyplot as plt

n =2
m=1000
component=1

X = pca.generateRandom2DData(n,m)

pca.plotOrigionalData(X)
X_Center, eig_vals, eig_vecs= pca.performPca(X)
sorted_eigenvectors= pca.sortEigenVectors(eig_vals,eig_vecs)
z = pca.reduceDimension(X_Center, sorted_eigenvectors, component)
pca.plotData(X,z,X,twoDplot = False)

X_recov = pca.recoverX(z, sorted_eigenvectors, component)

pca.plotData(X_Center,z,X_recov,twoDplot = True)

pca.show()