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
    node_ids = {}
    for i in range(len(PotentialsList)):
        node_name = PotentialsList[i].node.name
        node_ids[node_name] = i
    parents_list = [None]*len(node_ids)
    children_list = [[] for i in range(len(node_ids))]
    for Potential in PotentialsList:
        parents = [node_ids[parent.name] for parent in Potential.othernodes]
        current_id = node_ids[Potential.node.name]
        parents_list[current_id] = parents
        for parent in parents:
            children_list[parent].append(current_id)

    ordered_vars = []
    parents_dynamic = parents_list[:]
    free_nodes = {node_ids[Potential.node.name] for Potential in PotentialList 
                  if not Potential.othernodes}
    while free_nodes:
        free_node = free_nodes.pop()
        for child in children_list[free_node]:
            parents_dynamic[child].remove(free_node)
            if not parents_dynamic[child]:
                free_nodes.add(child)
        ordered_vars.append(free_node)
    return ordered_vars
        
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
    distribution = []
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
    val = 0
    return val

if __name__ == "__main__":
    NodeList, PotentialList = netlog('./asia.net')
    # for Node in NodeList:
    #     print(Node.name)
    #     print(Node.states)
    print(t_ordered_vars(PotentialList))