#!/usr/bin/env python
"""An object that provides various utilities for Gaussian random variables. Ben, 1 May 2008"""

import numpy as np
#import pylab 
from scipy.linalg import eigh
from scipy.linalg import solve
from scipy.linalg import inv
from scipy.linalg import det
from scipy import randn

class Gauss(object):
   """This object provides various utilities for Gaussian random variables. 
      The class can be initialized by either providing the mean and covariance, or 
      data from which the mean and covariance will be estimated. The data is assumed to
      a (d,n) array, where n is the number of data points and d the dimension of each data point, i.e. 
      each column of data corresponds to a d-dimensional observation, and there are n observations
      Caution: No test is provided to test whether data is in the correct shape.
 
     Example:
      -------
      >>>import gauss
      >>>#provide the mean and covariance
      >>>mean = [1.,1.]
      >>>cov  = [[2.,1.],[1.,3.]]
      >>>g    = gauss.Gauss(mean,cov)
      >>>g.image()
      >>>#provide data from which the mean and covariance are estimated   
      >>> g = gauss.Gauss(data=[[0.1,0.1,0.1],[0.,0.1,0.]])
      >>> g.mean
      array([ 0.1,0.03333333])
      >>> g.cov
      array([[ 0.015,  0.005],
       [ 0.005,  0.005]])
      ---------

      The class provides:
      eigvec        - Give the eigenvectors of cov
      eigval        - Give the eigenvalues of cov
      get_mean()    - Get the mean
      set_mean()    - Change the mean
      get_cov()     - Get the covariance
      set_cov()     - Change the covariance to a new value
      eval()        - Evaluate the Gaussian distribution
      sample()      - samples drawn from the distribution
      image()       - Display a 2d distribution
      plot()        - Plot a 1d distribution
      conditional() - Calculates a conditional distribution
      marginal()    - Calculates a marignal distribution
      bayes()       - Calculates the conditional distribution p(x|y) and marginal p(y)
   """

   def __init__(self, mean=None,cov=None,data=None):
      """A Gaussian random variable is defined by its mean and covariance.

      Parameters
      ----------
      Either specify data, or a mean and covariance. If the mean and covariance is estimated from data, each
      column of data corresponds to one d-dimensional observation, and there are n observations (columns). 

      mean : (d,) ndarray
          Mean of the distribution.
      cov : (d,d) ndarray
          Symmetric non-negative definite covariance matrix of the distribution.
      data : (n,d) ndarray
          If specified, the mean and covariance are calculated from these n, d-dimensional
          data vectors.

      """
      if not (mean is None and cov is None) and (not data is None):
          raise ValueError("Either provide data, or mean and covariance.")

      if data is None:  
         self.mean = np.asarray(mean).flatten()
         self.cov  = np.asarray(cov)
      else:
         data = np.asarray(data).transpose()    
         d,m  = data.shape                              # Assume that m is the number of data points   
         if m < d:
            raise ValueError('The number of data points is less than the dimension of the data')
         self.mean = np.mean(data,axis=1)               # Calculate the mean from the data
         self.cov  = np.cov(data)                       # Calculate the unbiased covariance from the data
 
      self._gamma = []
      self._x = []

   def get_mean(self):
      return self._mean

   def set_mean(self,mean):      
         self._mean = np.array(mean,ndmin=1,copy=False)

   mean = property(fget=get_mean, fset=set_mean)

   def get_cov(self):
      return self._cov

   def set_cov(self,cov):
      cov       = np.array(cov,ndmin=2)      
      rows,cols = cov.shape
      length    = np.size(self.mean)
    
      if not rows   == cols:
         raise ValueError('Covariance matrix must be square')

      if not length == rows:
         raise ValueError('The dimensions of the mean and covariance need to be the same')

      eigval, eigvec = eigh(cov)

      if np.min(eigval) < 0.:
         raise ValueError('The covariance is not positive semi-definite')
  
      self._eigval = eigval
      self._eigvec = eigvec  
      self._cov    = cov
      
   cov = property(fget=get_cov, fset=set_cov)

   @property
   def eigval(self):
      return self._eigval
 
   @property
   def eigvec(self):
      return self._eigvec

   def eval(self,x):
      """ Evaluate the Gaussian distribution at the value x
          Parameters
          ----------
          Provide the value where the Gaussian needs to be evaluated.
          x - (d,) ndarray, 

          Return
          ------
          The value of the Gaussian at x
      """
      x      = np.array(x).flatten()    
      mean   = self.mean.flatten()
      cov    = self.cov
      discr = np.dot(x-mean,solve(cov,x-mean))
      p = np.exp(-0.5*discr)/np.sqrt(det(2.*np.pi*cov))
      return p


   def sample(self, number_samples=1):
      """              sample(number_samples=1)
         Draw a sample from a d-dimensional normal distribution.
         Input:  number_samples - The number of samples that need to be generated
         Output: x - A dxn array containing the random samples 
      ToDo: Calculate directly using  numpy.random.multivariate_normal(mean,cov,number_samples)      
      """
      mean   = self.mean
      cov    = self.cov
      eigval = self.eigval
      eigvec = self.eigvec
      d      = len(mean)
      x      = np.zeros([d,number_samples])
      x      = np.array(x,ndmin=2)
      
      sqrt_eigval = np.sqrt(eigval)
      #Note: This can be directly calculated using numpy.random.multivariate_normal(mean,cov,number_samples)
      #One can also use Cholesky factorization instead of the SVD
      for j in range(number_samples):
         vec = np.dot(eigvec,randn(d)*sqrt_eigval)
         vec.flatten().shape,mean.flatten().shape
         x[:,j] = vec.flatten()+mean.flatten()
      x      = x.transpose()
      return x
   
   def image(self):
      """Display an image of the Gaussian distribution. This only accepts 2d distributions.
      """
      mean = self.mean
      cov  = self.cov
      d    = mean.size

      if d<>2:
         raise ValueError('Not an 2d array!')
       

      eigval = self.eigval
      max_eigval = np.max(eigval)

      xmin = mean[0]-2.*max_eigval
      xmax = mean[0]+2.*max_eigval
      ymin = mean[1]-2.*max_eigval
      ymax = mean[1]+2.*max_eigval
  
      x    = np.linspace(xmin,xmax,100) - mean[0]
      y    = np.linspace(ymin,ymax,100) - mean[1]

      # Calculate the image values from the mean and covariance.
      im   = np.zeros([np.size(x),np.size(y)])

      for i in range(len(x)):
         for j in range(len(y)):
            xi      = np.array([x[i],y[j]])
            im[j,i] = np.exp(-0.5*np.dot(xi,solve(cov,xi)))/np.sqrt(det(2.*np.pi*cov))
#      pylab.imshow(pylab.flipud(im),extent=(xmin,xmax,ymin,ymax))
#      pylab.show()
     
   def plot(self):
      """Plot a 1d Gaussian distribution. This only accepts 1d distributions.
      """
      mean = self.mean.flatten()
      cov  = self.cov.flatten()
            
      try:
         x = np.linspace(mean-3.*np.sqrt(cov),mean+3.*np.sqrt(cov),101)
         y = np.exp(-0.5*(x-mean)**2/cov)/np.sqrt(2.*np.pi*cov)
#         pylab.plot(x,y)
#         pylab.show()
      except:
         raise ValueError('Not a scalar variable')
      
   def conditional(self,xb):
      """         conditional(self,xb)  
         Calculates the mean and covariance of the conditional distribution
         when the variables xb are observed.
         Input: xb - The observed variables. It is assumed that the observed variables occupy the 
                     last positions in the array of random variables, i.e. if x is the random variable
                     associated with the object, then it is partioned as x = [xa,xb]^T.
         Output: mean_a_given_b - mean of the conditional distribution
                 cov_a_given_b  - covariance of the conditional distribution
      """
      xb         = np.array(xb,ndmin=1)
      nb         = len(xb)
      n_rand_var = len(self.mean)
      if nb >= n_rand_var:
         raise ValueError('The conditional vector should be smaller than the random variable!')
      mean = self.mean
      cov  = self.cov          

      # Partition the mean and covariance  
      na     = n_rand_var - nb
      mean_a = self.mean[:na]
      mean_b = self.mean[na:]
      cov_a  = self.cov[:na,:na]
      cov_b  = self.cov[na:,na:]
      cov_ab = self.cov[:na,na:]
      
      #Calculate the conditional mean and covariance
      mean_a_given_b = mean_a.flatten() + np.dot(cov_ab,solve(cov_b,xb.flatten()-mean_b.flatten()))
      cov_a_given_b  = cov_a - np.dot(cov_ab,solve(cov_b,cov_ab.transpose()))

      return mean_a_given_b, cov_a_given_b
         
   def marginal(self,d):
      """                 marginal(self,d)
         Marginalize over the last d elements of the random vector. This amounts to extracting the 
         appropriate partitions from the mean and covariance
         Input: d - the number of variables over which the mariginalization is done. It is assumed 
                    that the mariginalization is required over the last d elements.
         Output: mean - The mean of the marginal (of dimention n-d)
                 cov  - The covariance of the marginal (of dimension (n-d)x(n-d))
      """
          
       
      n_rand_var = np.size(self.mean)
      if d >= n_rand_var:
         raise ValueError('The marginalization vector should be smaller than the random variable!')

      # Partition the mean and covariance  
      na   = n_rand_var - d
      mean = self.mean[:na]
      cov  = self.cov[:na,:na]
      return mean, cov
      
   def bayes(self,A,b,L,*y):
      """     bayes(self,A,b,L,y)
         Assuming that the present object presents a Gaussian distribution p(x),
         one can provide the Gaussian distribution p(y|x) in the form
                           p(y|x) = N(y|Ax+b,L)
         where Ax+b is the mean (as a function of x) and L is the covariance.
         This allows one to calculate the mariginal p(y) = int_x p(x,y)dx = int_x p(y|x)p(x)dx.
         If  a value of y is also given, one can use Bayes theorem to calculate
                       p(x|y) = p(y|x)p(x)/p(y).
         This method returns the means and covariances of both p(y), and p(x|y) if a value
         for y is specified.
         Input: A, b     - The parameters of the linear mean of p(y|x)
                L        - The covariance matrix of p(y|x) 
                y        - The value of y on which p(x|y) is conditioned. Optional, if omitted, 
                           p(y) is calculated
         Output: y_mean  - The mean of p(y)
                 y_cov   - The covariance of p(y)
                 xy_mean - The mean of p(x|y) (if requested)
                 xy_cov  - The covariance of p(x|y) (if requested)
      """
      A      = np.array(A)
      b      = np.array(b)
      L      = np.array(L)
      y      = np.array(y).flatten()

      x_mean = self.mean
      x_cov  = self.cov

      
      try:
         # Calculate the mean and covariance of p(y)
         y_mean = np.dot(A,x_mean) + b
         y_cov  = L + np.dot(A,np.dot(x_cov,A.transpose()))

         # Calculate the mean and covariance of p(x|y) if y is given
         if np.size(y) > 0:        

            # Calculate mean and covariance of p(x|y)
            xy_cov  = inv(inv(x_cov) + np.dot(A.transpose(),solve(L,A)))         
            xy_mean = np.dot(xy_cov,(np.dot(A.transpose(),solve(L,y-b)) + solve(x_cov,x_mean)))
            return y_mean, y_cov, xy_mean, xy_cov
         else: 
            return y_mean, y_cov
      except:
         raise ValueError('Check the consistency of the dimensions')    

   def reset_ess(self):
       self._gamma = []
       self._x = []

   def update_ess(self, gamma, x):
       self._gamma.append(gamma)
       self._x.append(x)

   def maximize_params(self):
       _x = np.array(self._x).T
       _gamma = np.array([self._gamma]).T
       _N = _gamma.sum()
       mu = np.dot(_x, _gamma)
       mu = mu/_N
       sigma = np.zeros(self.cov.shape)
       cov_prior = np.diag(0.01*np.ones(self.cov.shape[0]))
       for i in range(_x.shape[1]):
           dm = np.asarray([_x]).T[i] - mu
           sigma += _gamma[i]*np.outer(dm,dm)
       sigma /= _N
       sigma += cov_prior

       self.mean = mu
       self.cov = sigma
