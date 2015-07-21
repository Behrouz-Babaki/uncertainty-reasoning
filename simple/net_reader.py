#!/usr/bin/env python
# encoding: utf-8
"""
HUGIN (.net) Bayesian network format reader.

Author: Behrouz Babaki
Using code written by Michiel Derhaeg
"""
from __future__ import print_function

import sys
import re
import argparse
from bayes_net import Node, Potential, BayesNet, findnode

netregex = re.compile("net[\s\n]*{(\n|.)*}")
noderegex = re.compile("node\s+([_\w-]+)[\n\s]*{([^}]*)}")
potentialregex = re.compile("potential\s*\(\s*([_\w-]+)\s*(\|\s*([_\w-]+\s*)+)?\)[\n\s]*{([^}]*)}")
statementRegex = re.compile("([_\w-]+)[]\s\n]*=[\n\s]*([^;]+);")
wordRegex = re.compile("(\w+)")
GoodBoolStates = [["true","false"],["yes","no"],["y","n"], ["t","f"]]

def netlog(inputfilepath):
    inputfile = open(inputfilepath, "r")
    netcode = re.sub("%.*","",inputfile.read()).lower()
    NodeList = []
    PotentialList =  []
    for nodematch in noderegex.finditer(netcode):
        NodeList.append(parseNode(nodematch.group(1),nodematch.group(2)))
    for potentialmatch in potentialregex.finditer(netcode):
        PotentialList.append(parsePotential(potentialmatch.groups()))
    return (NodeList,PotentialList)

def parseNode(name,body):
    newnode = Node(name)
    for statementMatch in statementRegex.finditer(body):
        parseStatement(newnode, statementMatch.group(1), statementMatch.group(2))
    return newnode

def parseStatement(node, element, value):
    if element == "states":
        for match in wordRegex.finditer(value):
            node.states.append(match.group(1))

def parsePotential(groups):
    node = groups[0]
    othernodes = groups[1]
    body = groups[3]
    newpotential = Potential(node)
    if othernodes:
        for nodematch in re.finditer("([_\w-]+)",othernodes):
            newpotential.othernodes.append(nodematch.group(0))
    for statementMatch in statementRegex.finditer(body):
        parseDataStatement(newpotential,statementMatch.group(1),statementMatch.group(2))
    return newpotential

def parseDataStatement(potential,element,value):
    if element == "data":
        value = re.sub("\)\s+\(",")(",re.sub("[\n\r]","",value ))
        data = []
        value = re.sub("[)(]"," ",value)
        output = ""
        for char in value:
            if char == " ":
                if len(output):
                    data.append(output)
                output = ""
            else:
                output += char
        potential.data = data

def normalizeData(data,nrOfStates):
    if (len(data)):
        for j in range(0,len(data), nrOfStates):
            sumofdata = 0
            for i in range(j, nrOfStates+j):
                sumofdata += float(data[i])
            for i in range(j, nrOfStates+j):
                data[i] = float(data[i]) / sumofdata
                data = list(map(str,data))
    return data

def cartesian (lists):
    if lists == []: return [()]
    return [x + (y,) for x in cartesian(lists[:-1]) for y in lists[-1]]

if __name__ == "__main__":
    desc = "HUGIN (.net) Bayesian network format reader."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("input_file", type=str, help="Input .net file")
    args = parser.parse_args()

    nodes, potentials = netlog(args.input_file)
    my_bn = BayesNet(nodes, potentials)


    for Node in my_bn.nodes:
        print(Node.name)
        print(Node.states)

    for Potential in my_bn.potentials:
        print()
        print(Potential.node.name + ':')
        if len(Potential.othernodes) > 0:
            for node in Potential.othernodes:
                print (node.name, end='\t')
            print()
        print(Potential.data)
        print(Potential.cumulatives)

