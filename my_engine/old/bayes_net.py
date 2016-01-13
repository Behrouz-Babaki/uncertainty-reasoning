#!/usr/bin/env python
# encoding: utf-8
"""
A class for representing Bayesian networks

Author: Behrouz Babaki
"""

class Node():
    def __init__(self,_name):
        self.name = _name
        self.states = []
        self.potential = None
    def nameWithState(self):
        reverse = list(self.states)
        reverse.reverse()
        if self.states in GoodBoolStates:
            return [self.name, "\+" + self.name]
        else:
            reverse = list(self.states)
            reverse.reverse()
            if reverse in GoodBoolStates:
                return ["\+" + self.name, self.name]
            else:
                names = list(self.states)
                for i in range(0,len(names)):
                    names[i] = self.name + "_" + names[i]
                return names

class Potential():
    def __init__(self,node):
        self.node = node
        self.othernodes = []
        self.data = []
        self.cumulatives = []
    def dimension(self):
        return 1 + len(self.othernodes)

class BayesNet():
    def __init__(self, Node_List, Potential_List):
        self.nodes = Node_List
        self.potentials = Potential_List
        make_references(self.nodes, self.potentials)
        create_cumulatives(self.nodes, self.potentials)


def findnode(name,nodes):
    output = None
    for n in nodes:
        if n.name == name:
            output = n
            break
    return output

def get_cpt_entry(node, node_val, parent_vals):
    assert(len(parent_vals) == len(node.potential.othernodes))
    vals = parent_vals + [node_val]
    index = 0
    for i in range(len(vals)):
        index += vals[i] * node.potential.cumulatives[i]
    return float(node.potential.data[index])

def process_bn(nodes, potentials):
    make_references(nodes, potentials)
    create_cumulatives(nodes, potentials)

def make_references(nodes, potentials):
    for potential in potentials:
        potential.node = findnode(potential.node, nodes)
        potential.node.potential = potential
        othernodes = []
        for node in potential.othernodes:
            othernodes.append(findnode(node, nodes))
        potential.othernodes = othernodes

def create_cumulatives(nodes, potentials):
    for potential in potentials:
        num_paretns = len(potential.othernodes)
        cumuls = [0] * (num_paretns + 1)
        cumuls[-1] = 1
        for i in range(num_paretns-1, -1, -1):
            cumuls[i] = cumuls[i+1] * len(potential.othernodes[i].states)
        potential.cumulatives = cumuls
