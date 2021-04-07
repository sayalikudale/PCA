Report on the task of Principle component Analysis :
The purpose of performing this task was to get the first principal component of the two-dimensional points in a space. After principal component is computed then by using this information points are reconstructed in the two-dimensional space.
Followed below steps to perform this task in using python: 

Step 1:  Random generation of points
The matrix of 2-dimensional points is randomly generated by using numpy random function. While generating this matrix positive linearity between the x and y co-ordinate of the point This is because it will generate the positive covariance and will achieve the direction of graph mentioned in a question. This matrix is plotted using matplotlib pyplot library. Figure 1 shows the generated points.
 
		Figure 1: Original data

Step 2: PCA Algorithm 	
1.	Data standardization: 
To perform PCA first step is data standardization; mean is calculated in each dimension and it is substracted from original value to gather the data around 0.  Below formula is used.
    	X_Center= X- mean 
2.	Calculate covariance:	
From covariance we can understand how values of each dimensions are dependent on each other. For above task we receive the positive covariance; it means that when value of x dimension increases, value of y dimension also increases and vice versa. This covariance is calculated using numpy library.
3.	Calculate eigen values and eigen vectors:
Eigen values and Eigen vectors are calculated using the numpy library. These values will be used to calculate the PCA. 
4.	Sorting the Eigen Vectors:
Eigen vectors are sorted in descending order because most significant values should be taken as first component and so on. 
5.	Compute the principal component:
In this step most significant eigen values are taken and dimensions are reduced to form the principal component. The Figure 2 is the first principal component.
 
		Figure 2: The first principal component 

Step 3: Reconstruct the original values:
Matrix multiplication of centered matrix and principal component matrix is done to reconstruct the original data. 
 
		Figure 3: Reconstructed points after PCA
In the Figure 3 the original data is shown in blue color and reduced features after applying the PCA are denoted in black color. Data points in the black color are considered as the significant data points and other data points are removed. 

From performing this task understanding of PCA is clearer. It is always preferrable to have most important, concise and significant features rather than a greater number of unsignificant features. This technique can be used to perform the features reduction without losing the important information from the dataset. 

