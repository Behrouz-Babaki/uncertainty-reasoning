module ParseNet

export Node
export Potential
export parse_net

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

end
