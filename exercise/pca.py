import numpy as np

X = [[3, 4], [1, 7], [2, 4]]

X -= np.mean(X, axis = 0) # zero-center the data (important)
cov = np.dot(X.T, X) / X.shape[0] # get the data covariance matrix

U,S,V = np.linalg.svd(cov)
Xrot = np.dot(X, U) # decorrelate the data
covrot = np.dot(Xrot.T, Xrot) / Xrot.shape[0] # get the data covariance matrix

print "U=", U
print "S=", S
print "V=", V
print "X=", X
print "cov=", cov
print "Xrot=", Xrot
print "covrot=", covrot
print np.linalg.norm(X[0]), np.linalg.norm(Xrot[0])