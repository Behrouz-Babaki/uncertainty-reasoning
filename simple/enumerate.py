#!/usr/bin/env python
"""
The enumeration algorithm for answering queries on a Bayesian network

Author: Behrouz Babaki
"""
from net_reader import Node, Potential, netlog

def t_ordered_vars(PotentialsList):
    """ Returns a topological ordering of nodes in a Bayesian network
    :param PotentialsList: The list of CPTs
    :returns: The topologically ordered list of nodes
    """
    # TODO implement function
    return OrderedNodes

def enumeration_ask(query, evidence):
    """ 
  Artificial Intelligence A Modern Approach (3rd Edition): Figure 14.9, page
 525.<br>
 <br>
  
 <pre>
  function ENUMERATION-ASK(X, e, bn) returns a distribution over X
    inputs: X, the query variable
            e, observed values for variables E
            bn, a Bayes net with variables {X} &cup; E &cup; Y /* Y = hidden variables //
            
    Q(X) <- a distribution over X, initially empty
    for each value x<sub>i</sub> of X do
        Q(x<sub>i</sub>) <- ENUMERATE-ALL(bn.VARS, e<sub>x<sub>i</sub></sub>)
           where e<sub>x<sub>i</sub></sub> is e extended with X = x<sub>i</sub>
    return NORMALIZE(Q(X))
    :param query: the query variables
    :param evidence: the evidence variables
    :returns: a distribution over query variables
    """
    # TODO implement function
    return distribution

def enumerate_all(variables, evidence):
    """
  Artificial Intelligence A Modern Approach (3rd Edition): Figure 14.9, page
 525.<br>
 <br>

  function ENUMERATE-ALL(vars, e) returns a real number
    if EMPTY?(vars) then return 1.0
    Y <- FIRST(vars)
    if Y has value y in e
        then return P(y | parents(Y)) * ENUMERATE-ALL(REST(vars), e)
        else return &sum;<sub>y</sub> P(y | parents(Y)) * ENUMERATE-ALL(REST(vars), e<sub>y</sub>)
            where e<sub>y</sub> is e extended with Y = y
  </pre>
  
  Figure 14.9 The enumeration algorithm for answering queries on Bayesian
  networks. <br>
  <br>
 <b>Note:</b> The implementation has been extended to handle queries with
  multiple variables. <br>
    """
    # TODO implement function
    return val

if __name__ == "__main__":
    NodeList, PotentialList = netlog('./asia.net')
    for Node in NodeList:
        print(Node.name)
        print(Node.states)
