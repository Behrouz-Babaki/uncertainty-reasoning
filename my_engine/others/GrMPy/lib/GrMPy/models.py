# Copyright 2009 Almero Gouws, <14366037@sun.ac.za>
"""
This module provides classes that implement Markov random field and
Bayesian network objects.
"""
__docformat__ = 'restructuredtext'

import numpy as np
from numpy import NaN
from scipy import sparse
import graph
import cpds
import general
import inference
import pdb
import potentials

class model(object):
    """ A graphical model. """
    def __init__(self, node_sizes, discrete_nodes=None, continuous_nodes=None):
        self.node_sizes_unobserved = node_sizes.copy()
        self.node_sizes = node_sizes.copy()
        self.discrete_nodes = np.array(discrete_nodes)
        self.continuous_nodes = np.array(continuous_nodes)


    def sum_product(self, evidence=None):
        """
        Execute the propagation phase of the sum-product algortihm on this
        graphical model.
        """
        if evidence is not None:
            self.enter_evidence(evidence)
        loglik = self.inference_engine.sum_product(evidence)

        return loglik

    def max_sum(self, evidence=None):
        """
        Execute the propagation phase of the max-sum algortihm on this
        graphical model.
        """
        if evidence is not None:
            self.enter_evidence(evidence)
        mlc = self.inference_engine.max_sum(evidence)

        return mlc

    def marginal_nodes(self, query, maximize=False):
        """
        Marginalize a set of nodes out of a clique.

        Parameters
        ----------
        query: List
            A list of the indices of the nodes to marginalize onto. This set
            of nodes must be a subset of one of the triangulated cliques (exact)
            or input user cliques (approximate) within the inference inference_engine.
            Marginalizing onto a single node will always work, because every
            node is the subset of some clique.

        maximize: Bool
            This value is set to true if we wish to maximize instead of
            marginalize, and False otherwise.
        """
        m = self.inference_engine.marginal_nodes(query, maximize)

        return m

class mrf(model):
    """
    A markov random field.
    """
    def __init__(self, adj_mat, node_sizes, clqs, lattice=False):
        """
        Initializes MRF object.

        adj_mat: Numpy array or Scipy.sparse matrix
            A matrix defining the edges between nodes in the network. If
            graph[i, j] = 1 there exists a undirected edge from node i to j.

        node_sizes: List or Int
            A list of the possible number of values a discrete
            node can have. If node_sizes[i] = 2, then the discrete node i
            can have one of 2 possible values, such as True or False. If
            this parameter is passed as an integer, it indicates that all
            nodes have the size indicated by the integer.

        clqs: List of clique objects (cliques.py)
            A list of the cliques in the MRF.

        lattice: Bool
            Lattice is true if this MRF has a lattice graph structure, and
            false otherwise.
        """
        """Assign the input values to their respective internal data members"""
        self.lattice = lattice
        self.num_nodes = adj_mat.shape[0]
        self.cliques = clqs
        self.node_sizes = np.array(node_sizes)
        self.node_sizes_org = np.array(self.node_sizes)

        """Convert the graph to a sparse matrix"""
        if ((type(adj_mat) == type(np.matrix([0]))) or
           (type(adj_mat) == type(np.array([0])))):
            adj_mat = sparse.lil_matrix(adj_mat)

        """In an MRF, all edges are bi-directional"""
#        self.adj_mat = adj_mat - \
#                           sparse.lil_diags([sparse.extract_diagonal(\
#                               adj_mat)], [0], (adj_mat.shape[0], \
#                                                    adj_mat.shape[0]))\
#                                                    + adj_mat.T

        self.adj_mat = adj_mat

        """
        Obtain elimination order, which is just the input order in the case
        of a lattice.
        """
        if self.lattice == True:
            self.order = range(0, self.adj_mat.shape[0])
        else:
            self.order = graph.topological_sort(self.adj_mat)

    def enter_evidence(self, evidence):
        self.evidence = evidence

    def init_inference_engine(self, exact=True, max_iter=10):
        """
        Determine what type of inference inference_engine to create, and intialize it.

        Parameters
        ----------
        exact: Bool
            Exact is TRUE if the type of inference must be exact, therefore,
            using the junction tree algorithm. And exact is FALSE if the type
            of inference must be approximate, therefore, using the loopy belief
            algorithm.

        max_iter: Int
            If the type of inference is approximate, then this value is maximum
            number of iterations the loopy belief algorithm can execute.
        """
        if exact:
            if self.lattice:
                print 'WARNING: Exact inference on lattice graphs not recommended'
            self.inference_engine = inference.JTreeInferenceEngine(self)
        elif self.lattice:
            """
            This version of the approximate inference inference_engine has been
            optimized for lattices.
            """
            self.inference_engine = inference.belprop_mrf2_inf_engine(self, max_iter)
        else:
            self.inference_engine = inference.belprop_inf_engine(self, max_iter)

    def learn_params_mle(self, samples):
        """
        Maximum liklihood estimation (MLE) parameter learing for a MRF.

        Parameters
        ----------
        samples: List
            A list of fully observed samples for the spanning the total domain
            of this MRF. Where samples[i][n] is the i'th sample for node n.
        """
        samples = np.array(samples)
        """For every clique"""
        for clq in self.cliques:
            """Obtain the evidence that is within the domain of this clique"""
            local_samples = samples[:, clq.domain]
            
            """If there is evidence for this clique"""
            if len(local_samples.tolist()) != 0:
                """Compute the counts of the samples"""
                counts = general.compute_counts(local_samples, clq.pot.sizes)

                """Reshape the counts into a potentials lookup table"""
                clq.unobserved_pot.T =\
                        general.mk_stochastic(np.array(counts, dtype=float))
                clq.pot.T = clq.unobserved_pot.T.copy()
                

    def learn_params_EM(self, samples, max_iter=10, thresh=np.exp(-4), \
                        exact=True, inf_max_iter=10):
        """
        EM algorithm parameter learing for a MRF, accepts partially
        observed samples.

        Parameters
        ----------
        samples: List
            A list of partially observed samples for the spanning the total
            domain of this MRF. Where samples[i][n] is the i'th sample for
            node n. samples[i][n] can be [] if node n was not observed in the
            i'th sample.    
        """
        """Set all the cliques parameters to ones"""
        for i in range(0, len(self.cliques)):
            self.cliques[i].unobserved_pot.T = \
                            np.ones(self.cliques[i].unobserved_pot.T.shape)

        """Create data used in the EM algorithm"""
        loglik = 0
        prev_loglik = -1*np.Inf
        converged = False
        num_iter = 0

        """Init the training inference inference_engine for the new BNET"""
        self.init_inference_engine(exact, inf_max_iter)
       
        while ((not converged) and (num_iter < max_iter)):
            
            """Perform an EM iteration and gain the new log likelihood"""
            loglik = self.EM_step(samples)
  
            """Check for convergence"""
            delta_loglik = np.abs(loglik - prev_loglik)
            avg_loglik = np.nan_to_num((np.abs(loglik) + \
                                        np.abs(prev_loglik))/2)
            if (delta_loglik / avg_loglik) < thresh:
                 """Algorithm has converged"""
                 break
            prev_loglik = loglik
            
            """Increase the iteration counter"""
            num_iter = num_iter + 1
            
    def EM_step(self, samples):
        """
        Perform an expectation step and a maximization step of the EM
        algorithm.

        Parameters
        ----------
        samples: List
            A list of partially observed samples for the spanning the total
            domain of this MRF. Where samples[i][n] is the i'th sample for
            node n. samples[i][n] can be [] if node n was not observed in the
            i'th sample.       
        """
        """Reset every cliques's expected sufficient statistics"""
        for clique in self.cliques:
            clique.reset_ess()

        """
        Set the log liklihood to zero, and loop through every sample in the
        sample set.
        """
        loglik = 0
        for sample in samples:
            """Enter the sample as evidence into the inference inference_engine"""
            sample_loglik = self.sum_product(sample[:])
            loglik = loglik + sample_loglik

            """For every clique in the MRF"""
            for clique in self.cliques:
                """
                Perform a marginalization over the entire cliques domain.
                This will result in a marginal containing the information
                for any nodes that were unobserved in the last entered sample,
                and will remove the 'expected' values for nodes that have been
                observed. Therefore, we are determining probability of the
                hidden nodes given the observed nodes and the current
                model parameters.
                """
                expected_vals = self.inference_engine.marginal_nodes(clique.domain)

                """Update this cliques expected sufficient statistics"""
                clique.update_ess(sample[:], expected_vals, self.node_sizes)

        """Maximize the parameters"""
        for clique in self.cliques:
            clique.maximize_params()

        return loglik
       
    def __str__(self):
        """
        Prints the values of the various members of a mrf object
        to the console.
        """
        print 'Model Graph:\n', self.adj_mat
        print '\nNumber of nodes: \n', self.num_nodes
        print '\nNode Sizes: \n', self.node_sizes
        print '\nCliques: \n', self.cliques
        print '\nPotentialss: \n', self.pots
        print '\nLattice \n', self.lattice
        print '\nOrder: \n', self.order
        return ''

class bnet(model):
    """
    A Bayesian network object.
    """
    def __init__(self, adj_mat, node_sizes, discrete_nodes=None, continuous_nodes=None, node_cpds=None):
        """
        Initializes BNET object.

        adj_mat: Numpy array or Scipy.sparse matrix
            A matrix defining the edges between nodes in the network. If
            graph[i, j] = 1 there exists a directed edge from node i to j.

        node_sizes: List or Int
            A list of the possible number of values a discrete
            node can have. If node_sizes[i] = 2, then the discrete node i
            can have one of 2 possible values, such as True or False. If
            this parameter is passed as an integer, it indicates that all
            nodes have the size indicated by the integer.

        node_cpds: List of CPD objects (node_cpds.py)
            A list of the CPDs for each node in the BNET.

        TODO:
            - ability for different nodes to share the same CPD
        """
        """Set the data members to the input values"""
        super(bnet, self).__init__(node_sizes, discrete_nodes, continuous_nodes)
        
        self.adj_mat = adj_mat
        self.num_nodes = adj_mat.shape[0]
        self.node_sizes = node_sizes.copy()

        self.node_cpds_N = len(self.node_sizes)

        if node_cpds is not None:
            #self.node_cpds_N = len(set(node_cpds))  determine shared node_cpds using set
            assert len(node_cpds)==self.node_cpds_N,\
                "Number of node CPDs inconsistent with model"
            self.node_cpds = np.array(node_cpds)
        else:
            self.node_cpds = np.array([],dtype=object)

        self.node_cpds_meta = [None]*self.node_cpds_N

        """Convert the graph to a sparse matrix"""
        if ((type(adj_mat) == type(np.matrix([0]))) or
           (type(adj_mat) == type(np.array([0])))):
            adj_mat = sparse.lil_matrix(adj_mat)
            
        """Obtain topological order"""
        self.order = graph.topological_sort(self.adj_mat)

    def init_inference_engine(self, exact=True, max_iter=10):
        """
        Determine what type of inference inference_engine to create, and intialize it.

        Parameters
        ----------
        exact: Bool
            Exact is TRUE if the type of inference must be exact, therefore,
            using the junction tree algorithm. And exact is FALSE if the type
            of inference must be approximate, therefore, using the loopy belief
            algorithm.

        max_iter: Int
            If the type of inference is approximate, then this value is maximum
            number of iterations the loopy belief algorithm can execute.
        """
        if exact:
            self.inference_engine = inference.JTreeInferenceEngine(self)
        else:
            self.inference_engine = inference.belprop_inf_engine(self, \
                                                       max_iter=10)

    def enter_evidence(self, evidence):
        """ Enter evidence into the model

        Parameters
        ----------
        evidence: List
            A list of any observed evidence. If evidence[i] = None, then
            node i is unobserved (hidden node), else if evidence[i] =
            SomeValue then, node i has been observed as being SomeValue.

        """
        # Convert evidence Numpy array and store in object
        self.evidence = np.array(evidence)

        # Reset node sizes.
        self.node_sizes = self.node_sizes_unobserved.copy()

        # Set sizes at oberved nodes to 1.
        self.observed_domain = []

        for i in range(len(evidence)):
            if evidence[i] is not None:
                self.node_sizes[i] = 1 
                self.observed_domain.append(i)

        # Normally the inference engine will need to be updated
        # after evidence has been entered.
        if type(self.inference_engine) is inference.JTreeInferenceEngine:
            self.inference_engine.update()

    def init_cpds(self):
            self.node_cpds = np.empty(self.node_cpds_N, dtype=object)
            for i in range(0, self.num_nodes):
                """Create a blank CPD for the node"""
                family = graph.family(self.adj_mat, i)
                if i in self.continuous_nodes:
                    self.node_cpds[i] = node_cpds.GaussianCPD()
                else:
                    fam = np.hstack([i,graph.parents(self.adj_mat, i)])
                    fam_sizes = self.node_sizes[fam]
                    self.node_cpds[i] = cpds.TabularCPD(np.ones(fam_sizes))

    def learn_params_mle(self, samples):
        """
        Maximum liklihood estimation (MLE) parameter learing for a BNET.

        Parameters
        ----------
        samples: List
            A list of fully observed samples for the spanning the total domain
            of this BNET. Where samples[i][n] is the i'th sample for node n.
        """
        """Convert the samples list to an array"""
        samples = np.array(samples)
        
        if len(self.node_cpds) == 0:
            self.init_cpds()
            """For every node in the BNET"""
            for i in range(0, self.num_nodes):
                """Get the samples within this nodes CPDs domain"""
                family = graph.family(self.adj_mat, i)
                local_samples = samples[:, family]

                """Learn the node parameters"""
                if len(local_samples.tolist()) != 0:
                    self.node_cpds[i].learn_params_mle(local_samples)
        else:
            """For every node in the BNET"""
            for i in range(0, self.num_nodes):
                """Get the samples within this nodes CPDs domain"""
                family = graph.family(self.adj_mat, i)
                local_samples = samples[:, family]
                
                """Learn the node parameters"""
                if len(local_samples.tolist()) != 0:
                    self.node_cpds[i].learn_params_mle(local_samples)

    def learn_params_EM(self, samples, max_iter=10, thresh=np.exp(-4), \
                        exact=True, inf_max_iter=10):
        """
        EM algorithm parameter learing for a BNET, accepts partially
        observed samples.

        Parameters
        ----------
        samples: List
            A list of partially observed samples for the spanning the total
            domain of this BNET. Where samples[i][n] is the i'th sample for
            node n. samples[i][n] can be [] if node n was not observed in the
            i'th sample.    
        """
        """If the CPDs have not yet been defined, then create them"""
        if len(self.node_cpds) == 0:
            self.init_cpds()

        """Create data used in the EM algorithm"""
        loglik = 0
        prev_loglik = -1*np.Inf
        converged = False
        num_iter = 0

        while ((not converged) and (num_iter < max_iter)):
            """Perform an EM iteration and gain the new log likelihood"""
            loglik = self.EM_step(samples)
            print 'iteration: %d\nloglik: %f'%(num_iter,loglik)

            """Check for convergence"""
            delta_loglik = np.abs(loglik - prev_loglik)
            avg_loglik = np.nan_to_num((np.abs(loglik) + \
                                        np.abs(prev_loglik))/2)
            if delta_loglik / (avg_loglik or 1.0) < thresh:
                 """Algorithm has converged"""
                 break
            prev_loglik = loglik
            
            """Increase the iteration counter"""
            num_iter = num_iter + 1

    def EM_step(self, samples):
        """
        Perform an expectation step and a maximization step of the EM
        algorithm.

        Parameters
        ----------
        samples: List
            A list of partially observed samples for the spanning the total
            domain of this BNET. Where samples[i][n] is the i'th sample for
            node n. samples[i][n] can be [] if node n was not observed in the
            i'th sample.       
        """
        """Reset every CPD's expected sufficient statistics"""
        for cpd in self.node_cpds:
            cpd.reset_ess()

        """
        Set the log liklihood to zero, and loop through every sample in the
        sample set.
        """
        loglik = 0
        for sample in samples:
            """Enter the sample as evidence into the inference inference_engine"""
            self.enter_evidence(sample[:])
            sample_loglik = self.sum_product()
            loglik = loglik + sample_loglik

            """For every node in the BNET"""
            for i in range(0, self.num_nodes):
                """
                Perform a marginalization over the entire CPDs domain.
                This will result in a marginal containing the information
                for any nodes that were unobserved in the last entered sample,
                and will remove the 'expected' values for nodes that have been
                observed. Therefore, we are determining probability of the
                hidden nodes given the observed nodes and the current
                model parameters.
                """
                expected_vals = self.inference_engine.marginal_family(i)

                """Update this nodes CPD's expected sufficient statistics"""
                self.node_cpds[i].update_ess(sample[:], expected_vals,\
                                        i, self)

        """Perform maximization step"""
        for cpd in set(self.node_cpds):
            cpd.maximize_params()

        return loglik
            

    def node_cpds_are_initialized(self):
        return len(self.node_cpds) == self.num_nodes

    def initialize_node_cpds(self):
        """For every node in the BNET"""
        for i in range(0, self.num_nodes):
            """Create a blank CPD for the node"""
            if self.node_sizes[0][i] != np.inf:
                self.node_cpds.append(node_cpds.tabular_CPD(i, self.node_sizes, \
                                              self.adj_mat))
            else:
                 self.node_cpds.append(node_cpds.GaussianCPD(i, self.node_sizes, \
                                                  self.adj_mat))
    def reset_node_cpds(self):
        self.node_cpds = []
        self.initialize_node_cpds()

    def __str__(self):
        """
        Prints the values of the various members of a bnet object
        to the console.
        """
        print 'Adjacency matrix:\n', self.adj_mat
        print '\nNumber of nodes: \n', self.num_nodes
        print '\nNode Sizes: \n', self.node_sizes
        print '\CPDs: \n', self.node_cpds
        print '\nOrder: \n', self.order
        return ''

class DBN(model):
    """ Dynamic Bayesian network 

        References
        ----------
        .. [1] K.P. Murphy, "Dynamic bayesian networks: representation, inference and learning"

    """
    def __init__(self, intra, inter, node_sizes, discrete_nodes=None,
            continuous_nodes=None, node_cpds = None, inference_inference_engine = None):
        """ Initialize a dynamic Bayesian network.

        Parameters
        ----------
        intra: numpy array
        inter: numpy array
        node_sizes:
        node_cpds
        """

        # Node sizes are initialized and stored for the 1st and 2nd time slices
        super(DBN, self).__init__(np.tile(node_sizes,2), discrete_nodes, continuous_nodes)
        self.intra = intra
        self.intra_N = len(self.intra)

        self.inter = inter
        self.inter_N = self.inter.sum()

        self.inference_inference_engine = inference_inference_engine

        # Adjacency matrix
        self.adj_mat = np.vstack([
                np.hstack([self.intra, self.inter]),
                np.hstack([np.zeros([self.intra_N]*2),self.intra])
                ])

        # Equivalence class. 
        self.equiv_class1 = range(self.intra_N)
        self.equiv_class2 = self.equiv_class1[:]
        for i in range(self.intra_N):
            l1 = np.array([j+self.intra_N for j in graph.parents(self.adj_mat, i)])
            l2 = graph.parents(self.adj_mat, i+self.intra_N)
            if all(l1 != l2):
                # Node has non-isomorphic parents
                self.equiv_class2[i] = max(self.equiv_class2)+1
            #else:
                # Node has isomorphic parents

        # Number of CPDs in the DBN. This will be equal to the number of nodes
        # in the first time-slice plus the number of nodes that have links
        # the the next time-slice.
        self.node_cpds_N = self.intra_N + self.inter_N

        if node_cpds is not None:
            assert(len(node_cpds)==self.node_cpds_N)
            self.node_cpds = np.array(node_cpds)
        else:
            self.node_cpds = np.empty(self.node_cpds_N, dtype=object)

        self.node_cpds_meta = [None]*self.node_cpds_N

    def learn_params_EM(self, samples, max_iter=10, thresh=np.exp(-4),\
                        exact=True, inf_max_iter=10):
        """ EM for DBN """
        # Flatten list of samples
        _samples = [sum(sample,[]) for sample in samples]

        # Learn unrolled DBN
        self.inference_engine.unrolled_bn.learn_params_EM(
                _samples, max_iter, thresh, exact, inf_max_iter)

    def enter_evidence(self, sample):
        # Flatten list of evidence
        _sample = sum(sample,[])
        self.inference_engine.unrolled_bn.enter_evidence(_sample)
                    
    def unroll(self, T):
        """ Unroll the dynamic BN.
        
        Return the resulting static BN constituting T time slices.

        The result static BN consisting of S*T node where S is the number of nodes
        in a single time slice of the DBN.

        Parameters
        ----------
        T: Int 
            Number of time slices in the unrolled network.
        """

        # Total number of nodes in unrolled DBN
        S = self.intra_N
        S_2 = 2*S
        self.N = S*T
        self.nodeN = S*T
        self.num_nodes = self.N

        # Initialize the adjacency matrix for the unrolled DBN.
        adj_mat = np.zeros([self.N]*2)

        for t in range(0, T):
            adj_mat[t*S:(t+1)*S,t*S:(t+1)*S] = \
                    self.intra
            if t != T-1:
                adj_mat[t*S:(t+1)*S,(t+1)*S:(t+2)*S] =\
                        self.inter

        # Initialize the node_sizes for the unrolled DBN.
        node_sizes = np.tile(self.node_sizes[self.intra_N:],T)

        node_cpds = [self.node_cpds[i] for i in self.equiv_class1]
        discrete_nodes = self.discrete_nodes.tolist()
        continuous_nodes = self.continuous_nodes.tolist()
        for t in range(1,T):
            discrete_nodes.append(self.discrete_nodes+t*S)
            continuous_nodes.append(self.continuous_nodes+t*S)
            node_cpds.extend([self.node_cpds[i] for i in self.equiv_class2])
        discrete_nodes = np.hstack(discrete_nodes)
        continuous_nodes = np.hstack(continuous_nodes)

        return bnet(adj_mat, node_sizes, discrete_nodes, continuous_nodes, node_cpds)
