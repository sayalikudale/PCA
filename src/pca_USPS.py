""" This Class is the driver class to perform PCA on given dataset
@author Sayali Kudale
"""
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pca as pca

component=10  # PCA component
imageId=2998    #This is the row number of required image in given 3000 rows

""" 
This method to extract the images from dataset matrix 
one image is extracted and converted into 16x16 matrix and plotted on a graph
@param mat , imageId
mat: dataset matrix
imageID: the row number in a matrix 
returns: it plots the given image and returns the matrix 
 """ 
def loadAndPlotImage(mat, imageId):
    A=(mat['A'])
    A1=A[imageId]
    X=np.reshape(A1,(16,16))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.imshow(X,cmap='gray')
    return A

""" 
This method converts the given matrix into 3000,256 format
and extract the image at specified post and  plot that image dimension onto the graph
@param X_recov
X_recov : reduced matrix of image after applying PCA
returns: returns the image matrix in 16x16 dimension
 """ 
def plotReducedImage(X_recov,imageId):
    X_reshape=np.reshape(X_recov,(3000,256))
    A1=X_reshape[imageId]
    X=np.reshape(A1,(16,16))
    fig = plt.figure() 
    ax1 = fig.add_subplot(111)
    ax1.imshow(X,cmap='gray')
    return X_reshape

""" 
This method calculates the recosntruction error using mean square error method
@param X, X_reshape
X : origional matrix of 3000 X 256 dimension
X_reshape: reconstructed matrix of origional data
""" 
def calReconstructionError(X,X_reshape):
    mse = np.mean((X - X_reshape)**2)
    print("Reconstruction Error for PCA component {}  is {}" .format(component,mse))

# load the file
mat = scipy.io.loadmat('USPS')
X=loadAndPlotImage(mat,imageId)
X_Center, eig_vals, eig_vecs = pca.performPca(X)

sorted_eigenvectors= pca.sortEigenVectors(eig_vals,eig_vecs)

z = pca.reduceDimension(X_Center, sorted_eigenvectors, component)

X_recov = pca.recoverX(z, sorted_eigenvectors, component)   

X_reshape=plotReducedImage(X_recov,imageId)

calReconstructionError(X,X_reshape)

pca.show()
