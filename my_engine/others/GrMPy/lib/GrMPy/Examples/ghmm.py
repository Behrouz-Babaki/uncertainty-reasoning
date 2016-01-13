""" Hidden Markov model (HMM) with Gaussian observations

This example shows how a hidden Markov model can be implemented
as a dynamic Bayesian network.

The DBN is be unrolled for N time-slices and inference/learning 
is performed in the unrolled static network with condition
PDFs shared across time slices.

x1 _ x2 ... xN
 |    |      |
y1   y2 ... yN

"""

# Imports
import numpy as np
import models
import cpds
import inference
from gauss import Gauss 

# Handles for readability.
inf = np.inf

# HMM parameters
# prior
pi = np.array([0.6, 0.4])

# state transition matrix
A = np.array([[0.7, 0.3],
              [0.2, 0.8]])

# state emission probabilities
B = np.array([Gauss(mean=np.array([1.0, 2.0]),
                    cov=np.eye(2)),
              Gauss(mean=np.array([0.0, -1.0]),
                    cov=np.eye(2))
              ])
# DBN
intra = np.array([[0,1],[0,0]])  # Intra-slice dependencies
inter = np.array([[1,0],[0,0]])  # Inter-slice dependencies

node_sizes = np.array([2,inf])

discrete_nodes = [0]
continuous_nodes = [1]

node_cpds = [cpds.TabularCPD(pi),
            cpds.GaussianCPD(B),
             cpds.TabularCPD(A)]

dbn = models.DBN(intra, inter, node_sizes, discrete_nodes,
        continuous_nodes, node_cpds)

inference_engine = inference.JTreeUnrolledDBNInferenceEngine()
inference_engine.model = dbn

inference_engine.initialize(T=5)
dbn.inference_engine = inference_engine

# INERENCE
evidence = [[None,[1.0,2.0]]
            ,[None,[3.0,4.0]],[None,[5.0,6.0]],[None,[7.0,8.0]],[None,[9.0,10.0]]]
dbn.enter_evidence(evidence)
print "Likelihood of single sample: %f"%dbn.sum_product()

# LEARNING
samples = [[[None, [-0.9094,-3.3056]],
            [None, [ 2.7887, 2.3908]],
            [None, [ 1.0203, 1.5940]],
            [None, [-0.5349, 2.2214]],
            [None, [-0.3745, 1.1607]]],
           [[None, [ 0.7914, 2.7559]],
            [None, [ 0.3757,-2.3454]],
            [None, [ 2.4819, 2.0327]],
            [None, [ 2.8705, 0.7910]],
            [None, [ 0.2174, 1.2327]]]
          ]

print "\nEM parameter learning:"
dbn.learn_params_EM(samples,max_iter = 10)
print "\nPrior (pi):"
print dbn.node_cpds[0]
print "\nTransition matrx (A):"
print dbn.node_cpds[2]
print "\nEmission probabilities (B):"
print dbn.node_cpds[1]
