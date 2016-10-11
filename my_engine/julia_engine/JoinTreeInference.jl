module JoinTreeInference

export Node
export Potential
export parse_net
export create_moral_graph
export triangulate_graph
export Sepset
export create_junction_tree

using DataStructures

net_pattern = r"net[\s\n]*{(\n|.)*}"
node_pattern = r"node\s+([_\w-]+)[\n\s]*{([^}]*)}"
potential_pattern = r"potential\s*\(\s*([_\w-]+)\s*(\|\s*([_\w-]+\s*)+)?\)[\n\s]*{([^}]*)}"
statement_pattern = r"([_\w-]+)[]\s\n]*=[\n\s]*([^;]+);"
word_pattern = r"(\w+)"

type Node
    name::String
    states::Array{String}
end

type Potential
    node::String
    other_nodes::Array{String, 1}
    data::Array{Float64, 1}
end

type Sepset
    first::Int64
    second::Int64
    nodes::Set{String}
    mass::Int64
    cost::Int64
end

function parse_node(nodematch::RegexMatch)
    node_name = nodematch.captures[1]
    node_values = Array(String, 0)
    state_match = match(statement_pattern, nodematch[2])
    if state_match[1] == "states"
        for v in eachmatch(word_pattern, state_match[2])
            push!(node_values, v.match)
        end
    end
    return Node(node_name, node_values)
end

function parse_data_statement(value)
    value = replace(value, r"[\n\r]", "")
    value = replace(value, r"\)\s+\(", ")(")
    value = replace(value, r"[)(]", " ")
    data = Array(Float64, 0)
    output = ""
    for char in value
        if char == ' '
            if length(output) > 0
                push!(data, float(output))
            end
            output = ""
        else
            output = string(output, char)
        end
    end
    return data
end

function parse_potential(potentialmatch::RegexMatch)
    node = potentialmatch[1]
    other_nodes = potentialmatch[2]
    others = Array(String, 0)
    if other_nodes != nothing
        for n in eachmatch(r"([_\w-]+)", other_nodes)
            push!(others, n.match)
        end
    end
    body = potentialmatch[4]
    values = match(statement_pattern, body)
    if values[1] == "data"
        data = parse_data_statement(values[2])
    end
    return Potential(node, others, data)
end

function parse_net(net_fname)
    f = open(net_fname) 
    net_str = readstring(f)
    close(f)

    net_str = lowercase(replace(net_str, r"%.*", ""))
    node_list = Array(Node, 0)
    for nodematch in eachmatch(node_pattern, net_str)
        push!(node_list, parse_node(nodematch))
    end
    
    potential_list = Array(Potential, 0)
    for potential_match in eachmatch(potential_pattern, net_str)
        push!(potential_list, parse_potential(potential_match))
    end
    return node_list, potential_list
end


function create_moral_graph(node_list::Array{Node,1}, 
                potential_list::Array{Potential,1})
    a_list = Dict{String, Set{String}}()
    for n in node_list
        a_list[n.name] = Set{String}()
    end
    
    function add_edge(i, j)
        push!(a_list[i], j)
        push!(a_list[j], i)
    end
    
    for p in potential_list
        for o in p.other_nodes
            add_edge(p.node, o)
        end
        for i in 1:length(p.other_nodes)
            for j in (i+1):length(p.other_nodes)
                add_edge(p.other_nodes[i], p.other_nodes[j])
            end
        end
    end
    return a_list
end

function get_next_node(g::Dict{String,Set{String}}, 
    node_weights::Dict{String,Int})

    best_f = 0
    best_w = 0
    best_k = 0
    function update(f, w, k)
        best_f = f
        best_w = w
        best_k = k
    end

    nodes =  collect(keys(g))
    first = true
    for (k, node) in enumerate(nodes)
        nei = collect(g[node])
        f = 0
        w = node_weights[node]
        for i in 1:length(nei)
            w += node_weights[nei[i]]
            for j in (i+1):length(nei)
                if !(nei[i] in g[nei[j]])
                    f += 1
                end
            end
        end
        if first
            first = false
            update(f, w, k)
        else
            if (f < best_f) || (f == best_f && w < best_w)
                update(f, w, k)
            end
        end
    end
    return nodes[best_k]
end

function triangulate_graph(g::Dict{String,Set{String}}, 
    node_list::Array{Node, 1})
    g1 = deepcopy(g)
    g2 = deepcopy(g)    
    function add_edge(i, j)
        for g in (g1, g2)
            push!(g[i], j)
            push!(g[j], i)
        end
    end
    
    node_weights = Dict{String, Int}()
    for node in node_list
        node_weights[node.name] = length(node.states)
    end
    
    clusters = Array{Set{String}, 1}()
    function check_add(g::Dict{String, Set{String}}, node::String)
        cluster = copy(g[node])
        push!(cluster, node)
        subsumed = false
        for c in clusters
            if issubset(cluster, c)
                subsumed = true
                break
            end
        end
        if !subsumed
            push!(clusters, cluster)
        end
    end
    
    while !isempty(keys(g1))
        next_node = get_next_node(g1, node_weights)
        check_add(g1, next_node)
        nei = collect(g1[next_node])
        for i in 1:length(nei)
            for j in (i+1):length(nei)
                add_edge(nei[i], nei[j])
            end
            delete!(g1[nei[i]], next_node)
        end
        delete!(g1, next_node)
    end
    return g2, clusters
end

function triangulate_graph(node_list::Array{Node,1}, 
                potential_list::Array{Potential,1})
    mg = create_moral_graph(node_list, potential_list)
    return triangulate_graph(mg, node_list)
end

function create_sepsets(clusters, node_list)
    node_weights = Dict{String, Int64}()
    for node in node_list
        node_weights[node.name] = length(node.states)
    end
    n = length(clusters)
    weights = Array{Int64, 1}()
    for i in 1:n
        w = 1
        for v in clusters[i]
            w *= node_weights[v]
        end
        push!(weights, w)
    end
    
    sepsets = Array{Sepset, 1}()
    for i in 1:n
        for j in (i+1):n
            # creat a new sepset
            nodes = intersect(clusters[i], clusters[j])
            mass = length(nodes)
            cost = weights[i] + weights[j]
            push!(sepsets, Sepset(i, j, nodes, mass, cost))
        end
    end
    
    sepset_comp(x, y) = (x.mass > y.mass) || ((x.mass == y.mass) && (x.cost < y.cost))
    return sort(sepsets, lt = sepset_comp)
end

function create_junction_tree(clusters::Array{Set{String}, 1}, sepsets::Array{Sepset, 1})
    n = length(clusters)
    output_tree = Dict{Int, Set{Int}}()
    for i in 1:n
        output_tree[i] = Set{Int}()
    end
    
    tree = IntDisjointSets(n)
    num_edges = 0
    for sepset in sepsets
        if num_edges == n - 1
            break
        end
        if ! in_same_set(tree, sepset.first, sepset.second)
            union!(tree, sepset.first, sepset.second)
            push!(output_tree[sepset.first], sepset.second)
            push!(output_tree[sepset.second], sepset.first)
            num_edges += 1
        end
    end
    return output_tree
end

function create_junction_tree(clusters::Array{Set{String}, 1}, node_list::Array{Node, 1})
    sepsets = create_sepsets(clusters, node_list)
    return create_junction_tree(clusters, sepsets)
end

end
