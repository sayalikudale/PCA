""" 
This Class computes the PCA from scratch
@author Sayali Kudale 
"""
import numpy as np
import matplotlib.pyplot as plt
  
""" 
This method generates the random points 
@param n , m
n: # dimentions 
m: # points 
 """ 
def generateRandom2DData(n, m):
   random = np.random.RandomState(1)
   uniform=random.rand(2, 2)
   normal=random.randn(n, m)
   X = np.dot(uniform,normal ).T
   return X

""" 
This method plot the matrix on graph 
@param x
x: 2D matrix 
""" 
def plotOrigionalData(X):
    fig, ax = plt.subplots(figsize = (12, 8))  
    ax.scatter(X[:, 0], X[:, 1],facecolors = 'b', edgecolors = 'b')
    plt.title("Origional Data", fontsize = 15)
    plt.xlabel("X")
    plt.ylabel("Y") 


""" 
This method performs PCA algorithm
@param x
x: input matrix 
""" 
def performPca(X):
    # mean centering
    mean = np.mean(X, axis=0)
    X_Center = (X - mean)
    # covariance matrix 
    cov = np.cov(X_Center.T)
    # eigen vectors
    eig_vals, eig_vecs = np.linalg.eig(cov)
    return X_Center, eig_vals,eig_vecs
 

""" 
This method sorts the eigen vectors in increasing order
@param eig_vals, eig_vec 
eig_vals: eigen values  
eig_vec: eigen Vector  
""" 
def sortEigenVectors(eig_vals,eig_vec):
    sorted_index = np.argsort(eig_vals)[::-1]
    sorted_eigenvectors = eig_vec[:,sorted_index]
    return sorted_eigenvectors

""" 
This method plots the PCA data and reconstructed data
@param X, X_pca , X_recon, twoDplot
X: origional matrix 
X_pca: matrix after PCA computation  
X_recon : reconstructed matrix after using PCA
twoDplot : whether PCA plot or reconstructed matrix plot
""" 
def plotData(X,X_pca, X_recon,twoDplot=False):
    plt.figure(figsize = (12, 8))
    plt.xlabel('X')
    plt.ylabel('Y')
    
    if twoDplot == False:
        plt.title("First Principle Component")
        plt.plot(X_pca, np.zeros_like(X_pca),'o',color = 'red')
    else:
        plt.title("Reconstruction: Dimensionality Reduced")
        plt.xlabel('X')
        plt.ylabel('Y')
        plot = plt.scatter(X[:, 0], X[:, 1],facecolors = 'b', edgecolors = 'b',label = "Actual")
        plot = plt.scatter(X_recon[:, 0], X_recon[:, 1],facecolors = 'k', edgecolors = 'k',  label = 'Projected')
        plt.legend()


""" 
This method reduces the dimentions of origional data by using eigen vector information 
@param X, eig_vec , com
X: origional matrix 
eig_vec: eigen vector 
com : # of components 
""" 
def reduceDimension(X, eig_vec, com):  
    reduce_mat = eig_vec[:, :com]
    z = np.dot(X, reduce_mat) 
    return z


""" 
This method recovers the origional points using reduced dimention matrix 
@param z, eig_vec , com
z: reduced matrix
eig_vec: eigen vector 
com : # of components 
""" 
def recoverX(z, eig_vec, com):  
    reduce_mat = eig_vec[:, :com]
    X_rec = np.dot(z, reduce_mat.T)
    return X_rec


""" 
method to show the  plotted data 
""" 
def show():
    plt.show()
