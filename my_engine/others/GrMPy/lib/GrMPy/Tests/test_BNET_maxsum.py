# Author: Almero Gouws <14366037@sun.ac.za>
"""
This is a tutorial on how to create a Bayesian network, and perform
exact MAX-SUM inference on it.
"""
"""Import the required numerical modules"""
import numpy as np
from sprinkler_data import sprinkler_evidence, sprinkler_mpe

"""Import the GrMPy modules"""
import models
import inference
import cpds

def test_bnet_maxsum():
    """
    Testing: MAX-SUM on BNET
    This example is based on the lawn sprinkler example, and the Bayesian
    network has the following structure, with all edges directed downwards:

                            Cloudy - 0
                             /  \
                            /    \
                           /      \
                   Sprinkler - 1  Rainy - 2
                           \      /
                            \    /
                             \  /
                           Wet Grass -3                
    """
    """Assign a unique numerical identifier to each node"""
    C = 0
    S = 1
    R = 2
    W = 3

    """Assign the number of nodes in the graph"""
    nodes = 4

    """
    The graph structure is represented as a adjacency matrix, dag.
    If dag[i, j] = 1, then there exists a directed edge from node
    i and node j.
    """
    dag = np.zeros((nodes, nodes))
    dag[C, [R, S]] = 1
    dag[R, W] = 1
    dag[S, W] = 1

    """
    Define the size of each node, which is the number of different values a
    node could observed at. For example, if a node is either True of False,
    it has only 2 possible values it could be, therefore its size is 2. All
    the nodes in this graph has a size 2.
    """
    node_sizes = 2 * np.ones(nodes)

    """
    We now need to assign a conditional probability distribution to each
    node.
    """
    node_cpds = [[], [], [], []]

    """Define the CPD for node 0"""
    CPT = np.array([0.5, 0.5])
    node_cpds[C] = cpds.TabularCPD(CPT)

    """Define the CPD for node 1"""
    CPT = np.array([[0.8, 0.2], [0.2, 0.8]])
    node_cpds[R] = cpds.TabularCPD(CPT)

    """Define the CPD for node 2"""
    CPT = np.array([[0.5, 0.5], [0.9, 0.1]])
    node_cpds[S] = cpds.TabularCPD(CPT)

    """Define the CPD for node 3"""
    CPT = np.array([[[1, 0], [0.1, 0.9]], [[0.1, 0.9], [0.01, 0.99]]])
    node_cpds[W] = cpds.TabularCPD(CPT)

    """Create the Bayesian network"""
    net = models.bnet(dag, node_sizes, node_cpds=node_cpds)

    """
    Intialize the BNET's inference engine to use EXACT inference
    by setting exact=True.
    """
    net.init_inference_engine(exact=True)

    """Create and enter evidence ([] means that node is unobserved)"""
    all_ev = sprinkler_evidence();
    all_mpe = sprinkler_mpe();

    count = 0;
    errors = 0;
    for evidence in all_ev:
        """Execute the max-sum algorithm"""
        mlc = net.max_sum(evidence)
##        print mlc
        
        mpe = all_mpe[count]
        errors = errors + np.sum(np.array(mpe) - np.array(mlc))
       
        count = count + 1

    assert errors == 0            
