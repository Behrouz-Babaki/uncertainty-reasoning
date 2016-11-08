module FactorOperations

export Factor
export product
export marginalize
export divice

using JoinTreeInference: Node, Potential

type Factor
    nodes::Array{String, 1}
    cardinality::Array{Int, 1}
    stride::Array{Int, 1}
    data::Array{Float64, 1}
end
    
function Factor(nodes::Array{String, 1}, cardinality::Array{Int, 1}, data::Array{Float64, 1})
    n = length(nodes)
    stride = Array{Int, 1}(n)
    s = 1
    for i in 1:n
        stride[i] = s
        s *= cardinality[i]
    end
    return Factor(nodes, cardinality, stride, data)
end

function product(factors::Array{Factor, 1})
        num_factors = length(factors)
        vars = union([f.nodes for f in factors]...)
        var_ids = Dict(j=>i for (i,j) in enumerate(vars))


        n = length(vars)
        cardinality = Array{Int, 1}(n)
        stride = Array{Int, 1}(n)
        
        idmap = Array{Array{Int, 1}, 1}(num_factors)
        for (i, f) in enumerate(factors)
            nv = length(f.nodes)
            idmap[i] = fill(0, n)
            for (j, v) in enumerate(f.nodes)
                id = var_ids[v]
                cardinality[id] = f.cardinality[j]
                idmap[i][id] = j
            end
        end
        
        s = 1
        for i in 1:n
            stride[i] = s
            s *= cardinality[i]
        end

        assignments = fill(1, n)
        ind = fill(1, num_factors)
        num_rows = reduce(*, cardinality)
        data = fill(1.0, num_rows)
        for i in 1:num_rows
            for (j, f) in enumerate(factors)
                data[i] *= f.data[ind[j]]
            end
            for l in 1:n
                assignments[l] += 1
                if assignments[l] > cardinality[l]
                    assignments[l] = 1
                    for j in 1:num_factors
                        id = idmap[j][l]
                        if id > 0
                            ind[j] -= factors[j].stride[id] * (cardinality[l]-1)
                        end
                    end
                else
                    for j in 1:num_factors
                        id = idmap[j][l]
                        if id > 0
                            ind[j] += factors[j].stride[id]
                        end
                    end
                    break
                end
            end
        end
        return Factor(vars, cardinality, stride, data)
end

function marginalize(f::Factor, v::Set{String})
    fn = length(f.nodes)
    n = fn - length(v)
    
    vars = Array{String, 1}(n)
    cardinality = Array{Int, 1}(n)
    idmap = fill(0, fn)
    isin = fill(false, fn)

    j = 1
    for (i, var) in enumerate(f.nodes)
        if !(var in v)
            vars[j] = var
            cardinality[j] = f.cardinality[i]
            idmap[i] = j
            isin[i] = true
            j += 1
        end
    end
    
    stride = Array{Int, 1}(n)
    s = 1
    for i in 1:n
        stride[i] = s
        s *= cardinality[i]
    end

    
    data = fill(0.0, reduce(*, cardinality))
    num_rows = reduce(*, f.cardinality)
    assignments = fill(1, fn)
    ind = 1
    for i in 1:num_rows
        data[ind] += f.data[i]
        for (l, n) in enumerate(f.nodes)
            assignments[l] += 1
            if assignments[l] > f.cardinality[l]
                assignments[l] = 1
                if isin[l]
                    ind -= stride[idmap[l]] * (f.cardinality[l]-1)
                end
            else
                if isin[l]
                    ind += stride[idmap[l]]
                end
                break
            end
        end
    end
    return Factor(vars, cardinality, stride, data)        
end

function divide(f1::Factor, f2::Factor)
    d = Dict((j, i) for (i,j) in enumerate(f2.nodes))
    v = Set(f2.nodes)
    
    fn = length(f1.nodes)
    idmap = fill(0, fn)
    isin = fill(false, fn)
    for (i, n) in enumerate(f1.nodes)
        if n in v
            isin[i] = true
            idmap[i] = d[n]
        end
    end
    
    assignments = fill(1, fn)
    f = deepcopy(f1)
    num_rows = reduce(*, f.cardinality)
    ind = 1
    for i in 1:num_rows
        f.data[i] /= f2.data[ind]
        for (l, n) in enumerate(f1.nodes)
            assignments[l] += 1
            if assignments[l] > f.cardinality[l]
                assignments[l] = 1
                if isin[l]
                    ind -= f.stride[idmap[l]] * (f.cardinality[l]-1)
                end
            else
                if isin[l]
                    ind += f.stride[idmap[l]]
                end
                break
            end
        end
    end
    return f
end

end
