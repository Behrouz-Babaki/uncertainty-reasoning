#!/usr/bin/env python
import copy


# a potential has a set of variables and a CPT

variable_c = ('c', 2)
variable_s = ('s', 2)
variable_r = ('r', 2)
variable_w = ('w', 2)

pc_vars = (variable_c,)
pc_cpt = {
          (0,) : 0.5,
          (1,) : 0.5
          }
pot_c = (pc_vars, pc_cpt)

pr_vars = (variable_c, variable_r)
pr_cpt = {
          (0, 0) : 0.8,
          (0, 1) : 0.2,
          (1, 0) : 0.2,
          (1, 1) : 0.8
          }
pot_r = (pr_vars, pr_cpt)

ps_vars = (variable_c, variable_s)
ps_cpt = {
          (0, 0) : 0.5,
          (0, 1) : 0.5,
          (1, 0) : 0.9,
          (1, 1) : 0.1
          }
pot_s = (ps_vars, ps_cpt)

pw_vars = (variable_r, variable_s, variable_w)
pw_cpt = {
          (0, 0, 0): 1,
          (0, 0, 1): 0,
          (0, 1, 0): 0.1,
          (0, 1, 1): 0.9,
          (1, 0, 0): 0.1,
          (1, 0, 1): 0.9,
          (1, 1, 0): 0.01,
          (1, 1, 1): 0.99
          }
pot_w = (pw_vars, pw_cpt)

def get_var_names(var_list):
    return [v[0] for v in var_list]

def get_var_vals(var_list):
    return [v[1] for v in var_list]

def init_cpt(variables):
    var_vals = get_var_vals(variables)
    
    def init_recurse (loc, var_vals, indices, cpt):
        if (loc == len(indices)):
            cpt[tuple(indices)] = 0
            return 
        for i in range(var_vals[loc]):
            indices[loc] = i
            init_recurse(loc+1, var_vals, indices, cpt)
    
    cpt = {}
    init_recurse(0, var_vals, [-1]*len(var_vals), cpt)
    return cpt

# a number of operations are defined over potentials
# one is marginalization wrt a subset of variables
def marg(pot, subset, project=False) :
    if not project:
        subset = tuple(set(pot[0])-set(subset))
    else:
        subset = tuple(subset)

    num_vars = len(pot[0])
    ind = [0] * num_vars
    for i in range(num_vars):
        if (pot[0][i] in subset):
            ind[i] = 1
            
    assert (set(subset).issubset(pot[0]))

    p2_cpt = init_cpt(subset)
    for indices, p in pot[1].iteritems():
            indices2 = tuple([indices[i] for i in range(num_vars) if ind[i]])
            p2_cpt[indices2] += p
        
    return (subset, p2_cpt)

# another one is factor multiplication
def mult(pot1, pot2):
    p3_vars = tuple(set(pot1[0]).union(set(pot2[0])))
    num_vars = len(p3_vars)
    ind1 = [0] * len(p3_vars)
    ind2 = [0] * len(p3_vars)
    for i in range(num_vars):
        if p3_vars[i] in pot1[0]:
            ind1[i] = 1
        if p3_vars[i] in pot2[0]:
            ind2[i] = 1
        
    p3_cpt = init_cpt(p3_vars)
    for indices, p in p3_cpt.iteritems():
        indices1 = tuple([indices[i] for i in range(num_vars) if ind1[i]])
        indices2 = tuple([indices[i] for i in range(num_vars) if ind2[i]])
        p3_cpt[indices] = pot1[1][indices1] * pot2[1][indices2]
    return (p3_vars, p3_cpt)
    
print mult(pot_c, pot_w)
print marg(pot_w, (variable_r, variable_s), project=False)

def var_elim(var_set, pot_list):
    """ the variable elimination algorithm
    
    :param var_set: the set of query (remaining) variables
    :param pot_set: a list of potentials
    :returns: a set of potentials
    """
    init_vars = set()
    pots = []
    for pot in pot_list:
        init_vars = init_vars.union(pot[0])
    
    for pot in pot_list:
        pot2 = copy.deepcopy(pot)
        current_vars = set(pot2[0])
        if not current_vars.issubset(var_set):
            rem = var_set.intersection(current_vars)
            pot2 = marg(pot, rem, project=True)
        pots.append(pot2)
    return pots

def bucket_elim(var_list, pot_list):
    """ the bucket elimination algorithm
        
        :param var_list: the ordered list of buckets
        :param pot_list: a list of potentials
        :returns: a list of factors over the remaining variables
    """
    # TODO turns out that you don't need a set
    buckets = {}
    for var in var_list:
        buckets[var] = []

    pot_ind = [True] * len(pot_list)
    for var in var_list:
        for i in range(len(pot_list)):
            if var in pot_list[i][0] and pot_ind[i]:
                buckets[var].append(pot_list[i])
                pot_ind[i] = False
                
    # remember that there might be remaining potentials
    rest = [pot[i] for i in range(len(pot_list)) if pot_ind[i]]
    
    # the elimination phase
    for i in range(len(var_list)):
        current_var = var_list[i]
        current_bucket = buckets[current_var]
        # multiply the factors
        bucket_mult = current_bucket[0]
        for j in range(1, len(current_bucket)):
            bucket_mult = mult(bucket_mult, current_bucket[j])
        # marginalize
        pot_m = marg(bucket_mult, (current_var,))
        # now move this to another bucket or rest
        found = False
        for j in range(i+1, len(var_list)):
            if var_list[j] in pot_m[0]:
                buckets[var_list[j]].append(pot_m)
                found = True
        if not found:
            rest.append(pot_m)
            
    return rest

p_list = [pot_c, pot_r, pot_s]
v_set = {variable_c, variable_r}
p = var_elim(v_set, p_list)
print (p)